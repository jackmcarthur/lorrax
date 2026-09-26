"""Route selection for the charge-channel ζ-solve.

The charge ζ-solve has one route, the *replicated* rank-truncated
pseudo-inverse (``replicated_rank_truncate``).  A rank-deficient C_q factored
any other way returns a ζ that is far too large, V_q rebuilds to relF O(10)
instead of O(1e-15), and Σ comes out as nonsense while every stage still
reports success (a full MoS2 12x12 G0W0 ran to rc=0 with a QP gap of -161 eV).
So the resolver has two outcomes only: the replicated route, or a refusal
when one q-batch of it cannot be held on a device.

Pure decision logic: no GPU and no multi-process launch required.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
from jax.sharding import Mesh

from isdf.core import _resolve_solver_kind


def _mesh(px: int, py: int) -> Mesh:
    """A (px, py) mesh over the available (CPU) devices."""
    n = px * py
    devs = jax.devices()
    if len(devs) < n:
        pytest.skip(f"needs {n} devices, have {len(devs)}")
    return Mesh(np.array(devs[:n]).reshape(px, py), ['x', 'y'])


# nq * n_mu**2 * 16 bytes vs the 4 GiB default cap.
_UNDER_CAP = dict(nq=144, n_rmu=1200)      # 3.32 GiB
_OVER_CAP = dict(nq=144, n_rmu=2412)       # 13.4 GiB stack; ~90 MiB per q-batch
# Above the PER-Q-BATCH factor cap too: one (mu, mu) c128 matrix alone
# exceeds _REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 GiB, so the refusal — not a
# batch split — is the contract.  17000^2 * 16 B = 4.31 GiB.
_OVER_FACTOR_CAP = dict(nq=144, n_rmu=17000)


def test_auto_selects_the_replicated_rank_truncation():
    assert _resolve_solver_kind(0, "auto", **_UNDER_CAP) == \
        "replicated_rank_truncate"


def test_a_current_channel_resolves_to_the_hoisted_lu():
    assert _resolve_solver_kind(1, "auto", n_rmu=64) == "lu"


def test_an_explicit_kind_passes_through():
    assert _resolve_solver_kind(0, "replicated_rank_truncate",
                                **_OVER_FACTOR_CAP) == "replicated_rank_truncate"


def test_rank_truncate_refuses_rather_than_downgrading():
    # Only a fit whose SINGLE (mu, mu) factor exceeds the per-batch cap has
    # nowhere to go; it must refuse, never fall back to a factor without the
    # rank-truncation cure.
    with pytest.raises(ValueError, match="rank_truncate"):
        _resolve_solver_kind(0, "auto", **_OVER_FACTOR_CAP)


def test_rank_truncate_stack_over_cap_still_runs_per_q_batch():
    # Stack over the cap but one q-batch under it keeps the cure: the
    # replicated factor is allocated one q-batch at a time.
    assert _resolve_solver_kind(0, "auto", **_OVER_CAP) == \
        "replicated_rank_truncate"


# ---------------------------------------------------------------------------
#  The closure Arm B left open on the FactorToken
# ---------------------------------------------------------------------------

def test_a_factor_token_cannot_enter_a_jit():
    """The pytree DECLINE, pinned — which is what makes it safe to rely on.

    ``FactorToken`` is a frozen dataclass that is deliberately not
    registered as a pytree, so jax refuses it by name at the jit boundary
    instead of tracing a handle.  (Route G also refuses a token by name
    before any jit: ``isdf.zeta_mubatch.ZetaG``.)  The day somebody
    registers the token as a pytree "for convenience" the refusal becomes a
    silent trace of an opaque block-cyclic handle.

    RED ARM: register FactorToken as a pytree node and this raises nothing.
    """
    from distrib_la import FactorToken

    token = FactorToken(op="cholesky", backend="slate", mesh=_mesh(1, 1),
                        n=8, nbatch=2, _factor=object())
    with pytest.raises(TypeError):
        jax.jit(lambda L_q: L_q)(token)


# ─────────────────────────────────────────────────────────────────────────
# The rank-truncation GATE can fire where it is INSTALLED — including
# inside a shard_map, which is where the q-parallel charge factor runs it.
# ─────────────────────────────────────────────────────────────────────────

_GATE_PROBE = r"""
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from common.shard_map import shard_map
from common import rank_criterion
from isdf.core import _certify_the_cut

mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))

def drive(spectrum):
    rank_criterion.raise_if_pending(mode="off")
    def body(lam):
        keep = lam > (1e-10 * jnp.max(lam, axis=-1, keepdims=True))
        _certify_the_cut(lam, keep, where="unit shard_map probe",
                         kappa_certified=rank_criterion.KAPPA_CERTIFIED_GRAM,
                         rcond=1e-10)
        return jnp.sum(lam, axis=-1)
    fn = jax.jit(shard_map(body, mesh=mesh, in_specs=P(('x', 'y'), None),
                           out_specs=P(('x', 'y')), check_vma=False))
    jax.block_until_ready(fn(jnp.asarray(spectrum)))
    return rank_criterion.pending()

# BOUND at kappa_eff = 1e9, an order above the 1e8 certified ceiling.
bad = np.asarray([[1e-30] * 4 + [1e-9] + [1.0] * 3], dtype=np.float64)
found = drive(bad)
assert found, "the gate recorded NOTHING from inside a shard_map"
assert "unit shard_map probe" in found[0], found

# CONTROL: the same kernel, a cut that binds well inside the certified
# regime.  Silence here is what makes the arm above evidence.
good = np.asarray([[1e-30] * 4 + [1e-3] + [1.0] * 3], dtype=np.float64)
assert drive(good) == [], "the gate fired on a well-conditioned cut"
rank_criterion.raise_if_pending(mode="off")
print("GATE_PROBE_OK")
"""


def test_the_zeta_rank_gate_fires_inside_a_shard_map(tmp_path):
    """A gate that cannot fire at its call site is not a gate.

    ``_certify_the_cut`` records a firing through ``jax.debug.callback`` so
    ``gw_init`` can refuse at the next host seam.  On the q-parallel schedule
    ``_charge_factor_math`` — and therefore this gate — runs INSIDE a
    ``shard_map``, which is a different execution context for a host callback
    than a plain ``jit``.  ``_close_the_cut`` only reaches its callback under
    the non-default ``strict``, so that path was effectively unexercised;
    this one reaches its callback on the DEFAULT.  Drive it rather than hope.

    IN A SUBPROCESS, and the reason is worth stating because it is a real
    constraint on the gate.  MEASURED on this module (jax 0.9.1, GPU): with
    no CPU device in the JAX backend, ``jax.debug.print``, ``jax.debug.
    callback`` AND ``io_callback`` all raise "failed to find a local CPU
    device to place the inputs on".  A production run never sees that —
    ``runtime.initialize_communicator_stack`` sets ``JAX_PLATFORMS=
    "cuda,cpu"`` (``runtime/__init__.py:395``), so a CPU device is always
    present — but a pytest process that imports ``isdf.core`` without booting
    the runtime does.  So the probe runs under ``JAX_PLATFORMS=cpu``, the
    idiom the multi-device workers use for their device faces.
    """
    import os
    import subprocess
    import sys

    src = tmp_path / "gate_probe.py"
    src.write_text(_GATE_PROBE)
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    res = subprocess.run([sys.executable, str(src)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0 and "GATE_PROBE_OK" in res.stdout, (
        f"gate probe failed rc={res.returncode}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
