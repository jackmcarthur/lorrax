"""normal_eigh never compiles an unused gathered retry (small CPU-only cases)."""
import importlib
import os
from types import SimpleNamespace

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import NamedSharding, PartitionSpec as P

plan_owner = importlib.import_module("distrib_la.plan")


@pytest.mark.parametrize("op", ["normal_eigh", "eigh", "checked_eigh"])
@pytest.mark.parametrize("fits", [True, False])
def test_placement_only_prices_retries_that_the_operation_can_run(monkeypatch, op, fits):
    mesh = SimpleNamespace(shape={"x": 2, "y": 2})
    gathered_calls, stack_calls, provider_calls = [], [], []

    def gathered(*args):
        gathered_calls.append(args)
        return True

    def stack(*args):
        stack_calls.append(args)
        return 40 if fits else 80

    def provider():
        provider_calls.append(True)
        return 29

    monkeypatch.setattr(plan_owner, "_gathered_admitted", gathered)
    monkeypatch.setattr(plan_owner, "_stack_bytes", stack)
    monkeypatch.setattr(plan_owner, "mesh_platform", lambda _: "unmeasured")
    result = plan_owner.place(op, mesh, 4, 8, np.complex128, 50, "scan", provider)
    assert len(gathered_calls) == (0 if op == "normal_eigh" else 1)
    assert len(stack_calls) == 1  # The actual candidate is always priced.
    assert stack_calls[0][-1] is (op != "normal_eigh")
    assert result.route == (plan_owner.ROUTE_BATCH_RESHARD if fits else "scan")
    assert result.program_bytes == (40 if fits else 29)
    assert len(provider_calls) == (0 if fits else 1)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("requires four CPU devices")
    return jax.sharding.Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_normal_gathered_flag_has_identical_executable_and_price():
    """The removed query's boolean never entered the normal solve's HLO."""
    mesh = _mesh()
    spec = NamedSharding(mesh, P(None, "x", "y"))
    operand = jax.ShapeDtypeStruct((4, 8, 8), np.complex128, sharding=spec)
    programs = [plan_owner._reshard_stack_program(
        "normal_eigh", mesh, operand.shape, "complex128", 1, "all", None, flag)
        for flag in (False, True)]
    lowered = [program.lower(operand) for program in programs]
    assert str(lowered[0].compiler_ir("stablehlo")) == str(lowered[1].compiler_ir("stablehlo"))
    prices = [plan_owner._analysis_bytes(program.compile()) for program in lowered]
    assert prices[0] == prices[1]


def test_normal_execution_skips_retry_query_and_keeps_literal_singular_values(monkeypatch):
    import distrib_la

    mesh = _mesh()
    rng = np.random.default_rng(71)
    host = rng.normal(size=(4, 8, 8)) + 1j * rng.normal(size=(4, 8, 8))
    operand = jax.device_put(host, NamedSharding(mesh, P(None, "x", "y")))
    plan = distrib_la.plan("eigh", mesh, n=8, backend="off")

    def unused(*args):
        pytest.fail("normal_eigh queried an unreachable gathered retry")

    monkeypatch.setattr(plan_owner, "_gathered_admitted", unused)
    values, vectors = plan.reshard_stack("normal_eigh", operand)
    values, vectors = np.asarray(values), np.asarray(vectors)
    expected = np.linalg.svd(host, compute_uv=False)[:, ::-1]
    np.testing.assert_allclose(values, expected, rtol=2.e-13, atol=2.e-13)
    gram = host.conj().swapaxes(-1, -2) @ host
    np.testing.assert_allclose(gram @ vectors, vectors * values[:, None, :] ** 2,
                               rtol=3.e-13, atol=3.e-13)
