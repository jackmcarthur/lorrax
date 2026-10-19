"""The bank's dispatch refuses an argument JAX would move inside the call (response_bank.
_dispatch_layout): a host (numpy) or uncommitted array over DISPATCH_MOVE_BYTES, which an AOT
executable accepts and copies silently (a committed array in another sharding JAX refuses
itself). run30k restart, 2026-10-19: sample_dyson ran 300 s with SMs idle and PCIe RX ~2.7 GB/s."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw import response_bank as rb

N = 512                                   # (2, 512, 512): 4 MiB complex64 (x64 off) or 8 MiB complex128, over the 1 MiB allowance


@pytest.fixture(scope="module")
def program():
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    fn = jax.jit(lambda h, chi, s: h @ chi * s, in_shardings=(face, face, rep), out_shardings=face)
    h = jax.device_put(jnp.ones((2, N, N), jnp.complex128), face)
    s = jax.device_put(jnp.float64(2.0), rep)
    return fn.lower(h, h, s).compile(), h, s


def test_arguments_placed_as_compiled_pass_and_print_once(program, capsys, monkeypatch):
    executable, h, s = program
    monkeypatch.setattr(rb, "_DISPATCH_PRINTED", set())
    rows = rb._dispatch_layout(executable, (h, h, s), "sample_dyson")
    assert all(ok for *_, ok in rows)
    rb._dispatch_layout(executable, (h, h, s), "sample_dyson")
    out = capsys.readouterr().out
    assert out.count("[response dispatch] sample_dyson arg") == 3        # once, three leaves


def test_a_small_host_scalar_may_move(program):
    executable, h, _ = program
    rows = rb._dispatch_layout(executable, (h, h, np.float64(2.0)), "other")
    assert rows[2][-1] is False and rows[2][2] <= rb.DISPATCH_MOVE_BYTES


@pytest.mark.parametrize("make", [lambda: np.ones((2, N, N), np.complex128),
                                  lambda: jnp.ones((2, N, N), jnp.complex128)],
                         ids=["host_numpy", "uncommitted"])
def test_a_large_argument_jax_would_move_is_refused_by_name(program, make):
    executable, h, s = program
    chi = make()
    assert executable(h, chi, s).shape == (2, N, N)        # JAX itself accepts it silently
    with pytest.raises(RuntimeError, match="GATE response_dispatch_layout: sample_dyson: arg 1"):
        rb._dispatch_layout(executable, (h, chi, s), "sample_dyson")
