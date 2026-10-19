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
