"""Independent dense resolvent checks for the ladder bank's exact actions."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _actions():
    def scan(function, carry, _xs, *, length, unroll):
        values = []
        for _ in range(length):
            carry, value = function(carry, None)
            values.append(value)
        return carry, np.stack(values)

    source = Path(__file__).parents[1] / "src/bse/bse_w_exact.py"
    tree = ast.parse(source.read_text())
    names = {"_resolvent_response_actions", "_moment_response_actions"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"jnp": np, "jax": SimpleNamespace(lax=SimpleNamespace(
        scan=scan, cond=lambda condition, yes, no, *, operand:
        (yes if condition else no)(operand)))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return tuple(namespace[name] for name in
                 ("_resolvent_response_actions", "_moment_response_actions"))


def _dense_ladder():
    rng = np.random.default_rng(27)
    density = rng.normal(size=(4, 3)) * 0.15
    v = np.diag([0.6, 0.8, 1.1])
    ring = density @ v @ density.T
    direct = rng.normal(size=(4, 4)) * 0.015
    direct = direct @ direct.T
    coupling = 0.01 * np.ones((4, 4))
    a = np.diag([1.5, 1.9, 2.3, 2.8]) + ring - direct
    b = ring - coupling
    h = np.block([[a, b], [-b, -a]])
    assert np.linalg.eigvalsh(np.diag([1.] * 4 + [-1.] * 4) @ h).min() > 0
    # A complex transition-basis change preserves every response, but catches
    # an accidental adjoint in the complex-frequency derivative contraction.
    u = np.diag(np.exp(1j * np.array([0.1, 0.7, -0.3, 1.2])))
    transform = np.kron(np.eye(2), u)
    h = transform @ h @ transform.conj().T
    seed = transform @ np.vstack([density @ v, -density @ v])
    vertex = v @ density.T @ u.conj().T
    readout = np.hstack([vertex, vertex])
    return h, seed, readout, vertex


def test_exact_slope_zero_limit_evenness_and_ladder_moments():
    slope_actions, moment_actions = _actions()
    h, seed, readout, vertex = _dense_ladder()

    def pack(value):
        return value.reshape(2, 4, -1).transpose(0, 2, 1)

    def flat(value):
        return value.transpose(0, 2, 1).reshape(8, -1)

    def solve(rhs, z, *, reflected=False):
        answer = np.linalg.solve(z * np.eye(8) - h, flat(rhs))
        return pack(answer), 0., 1

    rhs = pack(seed)
    fields = moment_actions(lambda x: pack(h @ flat(x)), rhs)
    for order in range(4):
        actual = vertex @ fields[f"M{order}"].T
        expected = readout @ np.linalg.matrix_power(h, order) @ seed / 2
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
        if order in (0, 2):
            np.testing.assert_allclose(actual, 0, atol=2e-15)

    for z in (0j, 0.2j, 0.3 + 0.7j):
        r = np.linalg.inv(z * np.eye(8) - h)
        value, _, _ = solve(rhs, z)
        slope, reflected, residual, iterations = slope_actions(solve, rhs, z, value)
        actual = vertex @ (slope[0] + slope[1]).T
        expected = (readout @ r @ r @ r @ seed if z == 0
                    else -readout @ r @ r @ seed / (2 * z))
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
        np.testing.assert_allclose(vertex @ (reflected[0] + reflected[1]).T,
                                   readout @ r @ seed, atol=2e-15, rtol=2e-14)
        # Independent central difference in s, including s=0, checks the sign
        # and the factor 2z without differentiating the action expression.
        def at_s(s):
            return readout @ np.linalg.solve(np.sqrt(complex(s)) * np.eye(8) - h, seed)
        step = 1e-4
        finite_difference = (at_s(z*z + step) - at_s(z*z - step)) / (2 * step)
        np.testing.assert_allclose(actual, finite_difference, atol=2e-11, rtol=2e-8)
        assert residual == 0. and iterations == 1


def test_odd_response_is_exposed_instead_of_projected_away():
    slope_actions, moment_actions = _actions()
    h, seed, readout, vertex = _dense_ladder()
    # A deliberately wrong anti-resonant source must remain visible to the
    # bank's evenness/odd-moment gates, rather than being Hermitized here.
    seed[4:] *= -1
    rhs = seed.reshape(2, 4, -1).transpose(0, 2, 1)
    def flat(x):
        return x.transpose(0, 2, 1).reshape(8, -1)
    def pack(x):
        return x.reshape(2, 4, -1).transpose(0, 2, 1)
    def solve(b, z, *, reflected=False):
        return pack(np.linalg.solve(z * np.eye(8) - h, flat(b))), 0., 1
    fields = moment_actions(lambda x: pack(h @ flat(x)), rhs)
    assert np.linalg.norm(vertex @ fields["M0"].T) > 1e-3
    z = 0.3 + 0.7j
    value = solve(rhs, z)[0]
    _, reflected, _, _ = slope_actions(solve, rhs, z, value)
    assert np.linalg.norm(vertex @ (reflected.sum(axis=0) - value.sum(axis=0)).T) > 1e-3


def test_compiled_shared_gmres_actions_against_dense_reference():
    import __future__
    import pytest
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).parents[1]
    from distrib_la import lstsq_hessenberg_local
    namespace = {"jax": jax, "jnp": jnp, "_BLOCK_GMRES_CACHE": {},
                 "lstsq_hessenberg_local": lstsq_hessenberg_local}
    selections = {
        "src/bse/bse_feast.py": {"_apply_shifted_matvec", "_gmres_solve_core"},
        "src/bse/bse_w_exact.py": {"_resolvent_response_actions", "_moment_response_actions",
                                   "_get_block_gmres_solver", "_get_block_moment_actions",
                                   "_bind_tol"},
    }
    for relative, names in selections.items():
        tree = ast.parse((root / relative).read_text())
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    h, seed, readout, vertex = _dense_ladder()
    seed = np.pad(seed, ((0, 0), (0, 1)))  # a real zero-RHS padding column
    rhs = seed.reshape(2, 4, -1).transpose(0, 2, 1)
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.array([device]), ("unit",))
    sh = SimpleNamespace(X=NamedSharding(mesh, PartitionSpec()))
    def matvec(value, operator):
        return (operator @ value.reshape(8, -1)).reshape(value.shape)
    with jax.default_device(device):
        operand = (jnp.asarray(h),)
        rhs = jnp.asarray(rhs)
        diagonal = jnp.asarray(np.diag(h).reshape(2, 1, 4))
        solver = namespace["_get_block_gmres_solver"](
            matvec, sh, 12, 1e-12, rhs.dtype, with_slope=True)
        plain = namespace["_get_block_gmres_solver"](
            matvec, sh, 12, 1e-12, rhs.dtype)
        for z in (0j, 0.2j, 0.3 + 0.7j):
            value, residuals, iterations, fields = solver(
                rhs, diagonal, jnp.asarray(z), operand)
            old_value = plain(rhs, diagonal, jnp.asarray(z), operand)[0]
            np.testing.assert_allclose(value, old_value, atol=2e-14, rtol=2e-13)
            r = np.linalg.inv(z * np.eye(8) - h)
            expected_slope = (readout @ r @ r @ r @ seed if z == 0
                              else -readout @ r @ r @ seed / (2*z))
            np.testing.assert_allclose(vertex @ np.asarray(fields["dWc_ds"]).T,
                                       expected_slope, atol=2e-12, rtol=2e-11)
            np.testing.assert_allclose(vertex @ np.asarray(fields["Wc_minus_z"]).T,
                                       readout @ r @ seed, atol=2e-12, rtol=2e-11)
            assert np.max(residuals) < 1e-10 and np.max(iterations) < 12
            np.testing.assert_array_equal(np.asarray(fields["dWc_ds"])[-1], 0)
        moments = namespace["_get_block_moment_actions"](matvec, sh, rhs.dtype)
        for name, action in moments(rhs, operand).items():
            order = int(name[1])
            np.testing.assert_allclose(vertex @ np.asarray(action).T,
                readout @ np.linalg.matrix_power(h, order) @ seed / 2,
                atol=2e-14, rtol=2e-13)


def test_streamed_facade_releases_all_moments_after_first_sample():
    import __future__
    import builtins
    import weakref

    def imports(name, *args, **kwargs):
        if name == "runtime.padding":
            return SimpleNamespace(padded_axis=lambda n, *a, **k: SimpleNamespace(carrier=n))
        return builtins.__import__(name, *args, **kwargs)

    input_rows = []

    def accumulate(old, tile, start, count, size, _mesh):
        if old is None:
            old = np.zeros((size, size), complex)
            input_rows.append(weakref.ref(old))
        old[:, start:start+count] = tile[:, :count]
        return old

    def zero_padding(value, logical):
        assert np.all(value[..., logical:, :] == 0) and np.all(value[..., :, logical:] == 0)

    source = Path(__file__).parents[1] / "src/bse/w_ladder.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"compute_wc_qwedge", "_identity_probe_blocks"}]
    sym = SimpleNamespace(q_irr_kgrid_int=np.zeros((2, 3), int),
                          irr_idx_q=np.arange(2), sym_idx_q=np.zeros(2, int),
                          q_irr_full_idx=np.arange(2))
    load_policies = []
    def load(*_args, **kwargs):
        load_policies.append(kwargs)
        return {"V_q0": np.zeros((8, 8)), "n_rmu": 7, "W_q": np.ones((8, 8, 2, 1, 1))}

    namespace = {"__builtins__": dict(vars(builtins), __import__=imports), "np": np, "jnp": np,
        "jax": SimpleNamespace(lax=SimpleNamespace(with_sharding_constraint=lambda a, _s: a)),
        "P": lambda *args: args, "NamedSharding": lambda _mesh, spec: spec,
        "WLadderWedge": SimpleNamespace,
        "load_bse_data_from_restart_sharded": load,
        "_psi_at_rest_2d": lambda data, _mesh: data, "_symmetry_tables": lambda _path: sym,
        "_accumulate_columns": accumulate, "gather_to_host": lambda a: a,
        "_assert_pad_block_is_zero": zero_padding}
    moment_visits = []
    consumed = []

    def prepare_owned(data, include_w, _mesh, *, consume_W_q=False):
        assert include_w and consume_W_q and data['W_q'] is not None
        consumed.append(data)
        data['W_R'],data['W_q']=data['W_q'],None
        return data

    namespace['ensure_W_R'] = prepare_owned

    def sweep(_data, _mesh, qrows, zrows, **kwargs):
        assert _data is consumed[-1] and _data['W_q'] is None and _data['W_R'] is not None
        for iz, z in enumerate(zrows):
            for iq, q in enumerate(qrows):
                for start, count, probes in kwargs["probe_blocks_for_q"](iq, q):
                    body = np.zeros((8, 8), complex)
                    body[:7, :7] = (iq + 1) * (iz + 1)
                    value = body @ probes.T
                    fields = {"dWc_ds": 2*value, "Wc_minus_z": value}
                    if iz == 0:
                        fields.update({f"M{j}": (j+1)*value for j in range(4)})
                        moment_visits.append((iq, start))
                    kwargs["on_result"](iq, q, iz, z, start, count, value,
                        np.zeros(len(probes)), np.ones(len(probes), int), fields)
            if kwargs["on_z_done"] is not None:
                kwargs["on_z_done"](iz, z)

    namespace["sweep_q_wedge"] = sweep
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    prior_moments = []
    seen = []

    def consume(iz, _z, value, _res, _iters, fields):
        assert all(reference() is None for reference in input_rows)
        if iz == 0:
            prior_moments.extend(weakref.ref(fields[f"M{j}"]) for j in range(4))
        else:
            assert all(reference() is None for reference in prior_moments)
            assert not any(name.startswith("M") for name in fields)
        seen.append(value.copy())

    kwargs = dict(input_file="unused", gmres_tol=1e-6, gmres_max_iter=300,
                  probe_chunk=2, with_slope=True, with_moments=True)
    mesh = SimpleNamespace(devices=np.empty((2, 2)))
    result = namespace["compute_wc_qwedge"]("unused", [0j, 1j, 2j], mesh,
                                           on_z=consume, **kwargs)
    assert result.wc is None and result.response_fields is None
    assert len(moment_visits) == 2*4
    materialized = namespace["compute_wc_qwedge"]("unused", [0j, 1j, 2j], mesh, **kwargs)
    assert len(consumed)==2 and consumed[0] is not consumed[1]
    assert len(load_policies) == 2
    assert all(policy["inject_head"] is False and policy["inject_w_head"] is True
               and policy["q0_head"] == "diagonal" and policy["native_grid_only"] is True
               for policy in load_policies)
    np.testing.assert_array_equal(materialized.wc, np.stack(seen))
    for name in ("M0", "M1", "M2", "M3"):
        assert materialized.response_fields[name].shape == (2, 8, 8)
