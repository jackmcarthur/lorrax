"""The passivity gate's eigh checks what the verdict uses (gw.shared_pole_gates.PASSIVITY_CHECK_CUT)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from distrib_la._result_check import checked_eigh  # noqa: E402
from gw.shared_pole_gates import _passivity_response_checks  # noqa: E402
from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as GATES  # noqa: E402


def _damaged(a, **_):
    """cuSOLVERMp's near-zero-cluster defect: two cluster vectors (ascending columns 3, 5) mixed."""
    w, v = jnp.linalg.eigh(a)
    return w, v.at[:, 3].set((v[:, 3] + 0.3 * v[:, 5]) / np.sqrt(1.09))


def _eigh(full=False):
    def eigh(x, cut=None):        # the checked route's contract: poisoned eigenvalues on a refusal
        (values, _), _ = checked_eigh((_damaged,), x[0], cut=None if full else cut)
        return values[None], None
    return eigh


def _response(top, bottom=0.0, n=64, rank=16, seed=3):
    """A whitened response of rank 16 with spectrum in [bottom, top] (bottom < 0 inside the cluster)."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)) + 0j)
    w = np.zeros(n); w[-rank:] = np.linspace(0.1, top, rank); w[0] = bottom
    return jnp.asarray(((q * w) @ q.conj().T)[None])


@pytest.mark.parametrize("top,bottom,full,passed", [
    (0.5, 0.0, False, True),     # rank-deficient and passive: the cluster's vectors are not read
    (0.5, 0.0, True, False),     # the same under a full check: refused by the defect (NaN values)
    (1.5, 0.0, False, False),    # lambda_max > 1: a true violation still fails
    (0.5, -1e-2, False, False),  # lambda_min < 0 inside the unchecked cluster: still fails
])
def test_passivity_checks_what_its_verdict_uses(top, bottom, full, passed):
    out = jax.jit(lambda r: _passivity_response_checks(r, eigh=_eigh(full), gates=GATES))(_response(top, bottom))
    assert bool(out["passivity"][0]) == passed, {k: np.asarray(v) for k, v in out.items()}


def test_passivity_checks_stay_on_the_face():
    """The face route's checks hold faces only (no whole n x n per rank: run30k passivity_held held
    2.4 n^2 per rank at any P from ``response - adjoint(response)``) and equal the local values."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    mesh, n = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")), 64
    face = NamedSharding(mesh, P(None, "x", "y"))
    skew = np.random.default_rng(5).standard_normal((n, n)) * 1e-9
    a = _response(0.5) + 1j * jnp.asarray(skew)[None]        # anti-Hermitian part 1e-9
    local_eigh = lambda x, cut=None: (jnp.linalg.eigvalsh(x), None)
    run = lambda eigh: jax.jit(lambda r: _passivity_response_checks(r, eigh=eigh, gates=GATES))
    local, onface = run(local_eigh)(a), run(local_eigh)(jax.device_put(a, face))
    h = 0.5 * (a[0] + a[0].conj().T)
    direct = np.linalg.norm(a[0] - a[0].conj().T) / (2 * np.linalg.norm(h))
    np.testing.assert_allclose(np.asarray(onface["passivity_antihermitian_relative"]), [direct], rtol=1e-12)
    for key in ("passivity_min", "passivity_max", "passivity_antihermitian_relative"):
        np.testing.assert_allclose(np.asarray(onface[key]), np.asarray(local[key]), rtol=1e-13, atol=1e-15)
    # Per rank: the response's face and its Hermitian part, nothing whole. The eigh is the
    # whole-mesh provider's shape (face in, face out), as the run30k census stands it in.
    keep = lambda x, cut=None: (jnp.zeros(x.shape[:-1]), jax.lax.with_sharding_constraint(x * 1, face))
    compiled = run(keep).lower(jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=face)).compile()
    whole = f"c128[1,{n},{n}]"
    assert not [line for line in compiled.as_text().splitlines()
                if "all-gather" in line and whole in line.replace(" ", "")]
    assert compiled.memory_analysis().temp_size_in_bytes <= 4 * 16 * n * n // mesh.size
