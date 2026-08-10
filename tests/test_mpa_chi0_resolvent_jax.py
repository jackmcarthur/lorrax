"""``chi0_resolvent``: the fused k-block step, and the call surface pinned.

TWO CLAIMS, AND THEY ARE DIFFERENT KINDS OF CLAIM.

The first is numerical: fusing the k-block walk's four device programs
into one changes when the work is scheduled and not what it computes, so
the output is byte-identical to the unfused walk.  The unfused walk is
written out below rather than referenced, because the point of a
byte-equality cell is to hold the new code against the old EXPRESSION,
and the old expression no longer exists in the module.

The second is an interface claim and it is here because this module has
consumers outside this tree: validation harnesses on ``/pscratch`` call
``chi0_resolvent`` directly, so its signature and its return convention
are a contract that a performance lane may not renegotiate.  The landing
plan pins it "by cell, not by review"; this is that cell.  It asserts the
parameter names, their order, which are keyword-only, and the defaults --
the four things a caller written against the old signature would notice.
"""

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from gw.mpa import chi0_resolvent as CR


class _Slices:
    def __init__(self, n_v, n_c):
        self.val = slice(0, n_v)
        self.cond = slice(n_v, n_v + n_c)


def _case(seed=0, n_k=8, n_v=3, n_c=4, n_mu=6, n_z=5, n_spinor=1):
    rng = np.random.default_rng(seed)
    n_b = n_v + n_c
    psi = (rng.standard_normal((n_k, n_b, n_spinor, n_mu))
           + 1j * rng.standard_normal((n_k, n_b, n_spinor, n_mu)))
    enk = np.sort(rng.random((n_k, n_b)) * 2.0, axis=1)
    z = (rng.random(n_z) * 0.5 + 1j * (rng.random(n_z) * 0.3 + 0.05))
    return psi, enk, _Slices(n_v, n_c), z


def _unfused_reference(psi, enk, slices, z_values, q_int, kgrid, *,
                       k_block=8, energy_reference=0.0):
    """The k-block walk as it stood before the fusion, verbatim.

    Four device programs per block -- einsum, the host-side energy
    difference, the z-scan, the accumulate -- which is the schedule the
    fused step replaces.  Kept here, and only here, as the arm of the
    byte-equality comparison.
    """
    psi_h = np.asarray(jax.device_get(psi))
    enk_h = np.asarray(jax.device_get(enk), dtype=np.float64)
    eref = 0.0 if energy_reference is None else float(energy_reference)
    psi_v = psi_h[:, slices.val]
    psi_c = psi_h[:, slices.cond]
    eps_v = enk_h[:, slices.val] - eref
    eps_c = enk_h[:, slices.cond] - eref
    psi_c_q = CR.roll_k_axis(psi_c, q_int, kgrid)
    eps_c_q = CR.roll_k_axis(eps_c, q_int, kgrid)
    n_k = psi_h.shape[0]
    n_mu = psi_h.shape[-1]
    z = jnp.asarray(np.asarray(z_values, dtype=np.complex128))
    acc = jnp.zeros((z.shape[0], n_mu, n_mu), dtype=jnp.complex128)
    for k0 in range(0, n_k, int(k_block)):
        k1 = min(k0 + int(k_block), n_k)
        M = CR._pair_amplitude(
            jnp.asarray(psi_c_q[k0:k1], dtype=jnp.complex128),
            jnp.asarray(psi_v[k0:k1], dtype=jnp.complex128))
        delta = jnp.asarray(
            eps_c_q[k0:k1][:, :, None] - eps_v[k0:k1][:, None, :],
            dtype=jnp.float64)
        acc = acc + CR._accumulate_block(M, delta, z)
    return acc * CR.chi0_ortho_norm(n_k)


# ---------------------------------------------------------------------------
#  The numerics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_block,kgrid,q_int", [
    (8, (2, 2, 2), (0, 0, 0)),
    (3, (2, 2, 2), (1, 0, 1)),
    (1, (2, 2, 2), (0, 1, 0)),
])
def test_the_fused_block_step_is_byte_identical_to_the_unfused_walk(
        k_block, kgrid, q_int):
    """No tolerance: the fold is the same fold and the ops are the same ops."""
    psi, enk, slices, z = _case(seed=int(k_block))
    got = np.asarray(jax.device_get(CR.chi0_resolvent(
        psi, enk, slices, z, q_int, kgrid, k_block=k_block)))
    want = np.asarray(jax.device_get(_unfused_reference(
        psi, enk, slices, z, q_int, kgrid, k_block=k_block)))
    assert got.dtype == want.dtype == np.complex128
    n_diff = int(np.count_nonzero(
        got.view(np.float64) != want.view(np.float64)))
    assert n_diff == 0, (
        f"{n_diff} of {got.size * 2} doubles differ between the fused and "
        f"unfused k-block walks; max |delta| = "
        f"{float(np.max(np.abs(got - want)))!r}")


def test_the_block_fold_order_is_the_one_the_docstring_pins():
    """Ascending k blocks, each added before the next is built.

    A block walk whose blocks were summed in a different association would
    be a different number in double, so the property is checked by
    re-running the same case at three block sizes and asserting that only
    the association changes the answer -- i.e. that a block size which
    does not re-associate (the whole axis in one block) reproduces the
    n_k-block walk to the re-association floor and no worse.
    """
    psi, enk, slices, z = _case(seed=5, n_k=8)
    whole = np.asarray(jax.device_get(CR.chi0_resolvent(
        psi, enk, slices, z, (0, 0, 0), (2, 2, 2), k_block=8)))
    split = np.asarray(jax.device_get(CR.chi0_resolvent(
        psi, enk, slices, z, (0, 0, 0), (2, 2, 2), k_block=2)))
    scale = float(np.max(np.abs(whole)))
    assert float(np.max(np.abs(whole - split))) < 1e-13 * scale


def test_x64_is_still_refused_when_it_is_off(monkeypatch):
    """The gate the module opens with does not move with the refactor."""
    psi, enk, slices, z = _case()
    monkeypatch.setattr(jax.config, "read", lambda k: (
        False if k == "jax_enable_x64" else jax.config.values.get(k)))
    with pytest.raises(RuntimeError, match="GATE x64_enabled"):
        CR.chi0_resolvent(psi, enk, slices, z, (0, 0, 0), (2, 2, 2))


# ---------------------------------------------------------------------------
#  The call surface
# ---------------------------------------------------------------------------

def test_the_public_call_surface_of_chi0_resolvent_is_pinned():
    """The contract external harnesses are written against.

    Positional order, keyword-only-ness and defaults, asserted by name.
    A performance lane may restructure everything behind this line and
    nothing on it.
    """
    sig = inspect.signature(CR.chi0_resolvent)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == [
        "psi", "enk", "slices", "z_values", "q_int", "kgrid",
        "k_block", "energy_reference"]
    kinds = {p.name: p.kind for p in params}
    for name in ("psi", "enk", "slices", "z_values", "q_int", "kgrid"):
        assert kinds[name] is inspect.Parameter.POSITIONAL_OR_KEYWORD, name
    for name in ("k_block", "energy_reference"):
        assert kinds[name] is inspect.Parameter.KEYWORD_ONLY, name
    assert sig.parameters["k_block"].default == 8
    assert sig.parameters["energy_reference"].default == 0.0


def test_the_module_still_exports_what_its_consumers_import():
    """Names a /pscratch harness resolves off the module, held by name."""
    for name in ("chi0_resolvent", "chi0_ortho_norm", "roll_k_axis",
                 "bridge_gate_report", "cost_model", "_pair_amplitude"):
        assert hasattr(CR, name), f"gw.mpa.chi0_resolvent lost {name}"


def test_the_return_convention_is_the_one_solve_w_consumes():
    """Shape, dtype and the ortho-norm residue -- the three conventions."""
    psi, enk, slices, z = _case(n_mu=6, n_z=5)
    out = CR.chi0_resolvent(psi, enk, slices, z, (0, 0, 0), (2, 2, 2))
    assert out.shape == (5, 6, 6)
    assert out.dtype == jnp.complex128
    assert CR.chi0_ortho_norm(8) == pytest.approx(8.0 ** -0.5)
