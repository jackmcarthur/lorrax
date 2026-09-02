"""The chi0/W two-role live-set defect, and its bounded-schedule fix.

THE DEFECT (KNOWN_LORRAX_ISSUES.md, "GN-PPM probe chi0 has no bounded
two-role live-set plan at 81 q", 2026-08-20).  ``gw.w_isdf.compute_chi0``'s
tau-scan needs an ``O(nq*mu^2/P)`` scratch arena that is the SAME size for
every role regardless of tau-node count (a ``lax.scan``'s compiled buffer
graph is trip-count-independent) and is legitimately unchunked over q -- the
flat-k FFT it runs needs the whole q/k axis local on every rank.  On the
production MoS2 9x9x1/P16 deck the static role's build completed and the
probe role's IDENTICAL build then OOM'd (RESOURCE_EXHAUSTED, exact request
27,262,284,032 B on two independent jobs) with the static role's completed W
STILL a live on-device array, held by nothing but a Python dict reference in
``gw.screening.compute_screening``.

THE FIX under test here has two layers:

  1. ``common.collectives.spill_to_host`` / ``restore_from_host`` -- a
     generic, zero-collective device<->host round trip for an
     already-sharded array (tested directly, on a real multi-device mesh,
     for bit-exactness and for actually freeing the device buffer).
  2. ``gw.screening.compute_screening`` routes every role's W through it:
     every role but the LAST is spilled the moment its own gate passes, and
     restored only after the whole role loop has run.  Tested here with
     tiny stand-in W's and ``compute_static_w``/``_gate_w`` monkeypatched
     out -- the assertion is purely about SCHEDULING (is the earlier role's
     buffer gone by the time the later role's build starts), independent of
     the chi0/W physics.

THE RED TWIN (``test_a_scheduler_that_never_spills_is_caught``) proves the
scheduling assertion can actually fail: it drives the exact same harness
against a stand-in ``compute_screening`` body that keeps every role
resident, and the fixture catches it.  A check that cannot fail certifies
nothing (QUALITY_PATTERNS' addendum) -- this is that check's own negative
control.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import collectives


# ---------------------------------------------------------------------------
# 1. The primitive: spill_to_host / restore_from_host
# ---------------------------------------------------------------------------

@pytest.mark.mesh(4)
def test_spill_to_host_round_trips_a_p_none_x_y_sharded_array_and_frees_it():
    """Bit-exact round trip, on the EXACT sharding chi0/V/W use in production.

    ``P(None, 'x', 'y')`` on a 2x2 mesh is the canonical sharding for every
    N_mu^2-class tensor on this seam (chi0_q, V_q, W_q) -- both diagnoses
    that motivated this fix agreed the sharding SHAPE was already correct;
    this proves the spill primitive preserves it exactly, not merely
    approximately.
    """
    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    shape = (3, 8, 8)
    rng = np.random.default_rng(0)
    host = (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)
    arr = jax.device_put(host, sharding)

    spill = collectives.spill_to_host(arr)

    # THE PROPERTY THAT MAKES THIS A LIVE-SET TOOL, NOT A HINT: the device
    # buffer is freed at this statement, not whenever Python's refcounter
    # gets around to it.
    assert arr.is_deleted()

    restored = collectives.restore_from_host(spill)
    assert restored.sharding == sharding
    assert tuple(restored.shape) == shape
    np.testing.assert_array_equal(np.asarray(jax.device_get(restored)), host)


@pytest.mark.mesh(4)
def test_spill_to_host_touches_no_collective_primitive():
    """Structural check that the "zero collectives" claim isn't just prose.

    ``spill_to_host``/``restore_from_host`` must not call any of the
    collective primitives this same module wraps elsewhere -- a spill that
    silently became an all-gather would defeat the whole point (the same
    trap ``device_put_process_local``'s docstring documents for
    ``jax.device_put`` on a host table).
    """
    import inspect

    src = inspect.getsource(collectives.spill_to_host) + inspect.getsource(
        collectives.restore_from_host)
    for banned in ("all_gather", "psum", "sync_global_devices",
                   "process_allgather", "all_to_all"):
        assert banned not in src, (
            f"spill/restore round trip mentions {banned!r} -- it must stay "
            "a per-process, no-collective operation")


# ---------------------------------------------------------------------------
# 2. The scheduling harness: drive a role loop and assert the live-set bound
# ---------------------------------------------------------------------------
#
# Shared between the real fix (test 2a) and the red twin (test 2b) so the
# ONLY thing that differs between "caught" and "not caught" is which
# ``_store_role``-shaped function is under test -- not the harness.

def _drive_two_role_schedule(store_role):
    """Run a 2-role (static, probe) loop through ``store_role`` and return
    the built-history of "was the static role's buffer still live when the
    probe role's build started".

    ``store_role(role, idx, n_roles, W_by_role, W) -> None`` must mutate
    ``W_by_role[role]`` however it likes (store live, spill, whatever it is
    being asked to prove).  Returns ``(W_by_role, static_was_live_at_probe)``.
    """
    n_roles = 2
    W_by_role: dict = {}
    static_w = jnp.eye(2, dtype=jnp.complex128)[None]           # tiny stand-in
    store_role("static", 0, n_roles, W_by_role, static_w)

    # The moment that matters: has the PROBE role's build started while the
    # STATIC role's W is still resident?  In production this is the instant
    # ``compute_chi0``'s tau-scan dispatches for the probe role; here it is
    # simply "after ``store_role`` ran for the static role, before the probe
    # role's own W is computed" -- which is exactly the window the real
    # ``compute_screening`` loop leaves between iterations.
    static_was_live_at_probe = not static_w.is_deleted()

    probe_w = jnp.eye(2, dtype=jnp.complex128)[None] * 2.0
    store_role("probe", 1, n_roles, W_by_role, probe_w)

    # Uniform restore pass, mirroring compute_screening's own tail.
    for role, val in list(W_by_role.items()):
        if isinstance(val, collectives.HostSpill):
            W_by_role[role] = collectives.restore_from_host(val)
    return W_by_role, static_was_live_at_probe


def _bounded_store_role(role, idx, n_roles, W_by_role, W):
    """The shape ``gw.screening.compute_screening._store_role`` takes:
    everything but the LAST role is spilled immediately."""
    if idx == n_roles - 1:
        W_by_role[role] = W
    else:
        W_by_role[role] = collectives.spill_to_host(W)


def test_the_bounded_schedule_frees_the_static_role_before_the_probe_role():
    """THE GREEN CASE.  Mirrors ``compute_screening``'s own ``_store_role``."""
    result, static_was_live_at_probe = _drive_two_role_schedule(
        _bounded_store_role)
    assert static_was_live_at_probe is False, (
        "the static role's W was still a live on-device array when the "
        "probe role's build started -- the two-role live-set bound is not "
        "holding")
    assert set(result) == {"static", "probe"}
    for w in result.values():
        assert not w.is_deleted()
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["static"])), np.eye(2)[None])
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["probe"])), 2.0 * np.eye(2)[None])


def test_a_scheduler_that_never_spills_is_caught():
    """THE RED TWIN.  A scheduler that keeps every role resident (the
    pre-fix shape: ``W_by_role[req.role] = W`` with no spill at all) must
    make the harness's own assertion fail -- otherwise the green test above
    is not actually checking anything."""
    def _unbounded_store_role(role, idx, n_roles, W_by_role, W):
        W_by_role[role] = W          # never spills -- the pre-fix behaviour

    with pytest.raises(AssertionError):
        result, static_was_live_at_probe = _drive_two_role_schedule(
            _unbounded_store_role)
        assert static_was_live_at_probe is False


# ---------------------------------------------------------------------------
# 3. The production seam: gw.screening.compute_screening itself
# ---------------------------------------------------------------------------

class _StubPPM:
    probe_chi_reuse = "off"


class _StubConfig:
    ppm = _StubPPM()
    minimax_config = None


class _StubMeta:
    kgrid = (1, 1, 1)
    nk_tot = 1


def test_compute_screening_spills_an_earlier_role_before_a_later_roles_build(
        monkeypatch):
    """Wires test 2's harness shape onto the REAL ``compute_screening`` loop.

    ``compute_static_w`` and ``_gate_w`` are monkeypatched out -- this test
    is not about chi0/W physics, it is about whether ``compute_screening``
    ACTUALLY calls the bounded schedule on the production role loop, not
    just that the schedule is correct in isolation (test 2 above).
    """
    from gw import screening
    from gw.screening import ScreeningRequest

    state: dict = {}
    calls: list = []

    def _fake_compute_static_w(wfns, V_q, quad, *, role, **kwargs):
        if role != "static":
            static_w = state["static_w"]
            assert static_w.is_deleted(), (
                "compute_screening still holds the static role's W as a "
                f"live on-device array when the {role!r} role's chi0/W "
                "build starts -- the two-role live-set bound regressed")
        w = jnp.eye(2, dtype=jnp.complex128)[None] * (1.0 if role == "static"
                                                        else 2.0)
        calls.append(role)
        if role == "static":
            state["static_w"] = w
        return w

    monkeypatch.setattr(screening, "compute_static_w", _fake_compute_static_w)
    monkeypatch.setattr(screening, "_gate_w", lambda *a, **k: None)
    monkeypatch.setattr(
        "gw.minimax_screening.build_imag_quadrature",
        lambda *a, **k: "unused-quad-stand-in")

    requests = [ScreeningRequest(0.0 + 0.0j, "static"),
                ScreeningRequest(1j * 1.0, "probe")]

    result = screening.compute_screening(
        wfns=None, V_q=None, requests=requests,
        quad=None, e_ref=0.0,
        sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
        config=_StubConfig(), meta=_StubMeta(), mesh_xy=None,
        print_fn=lambda *a, **k: None)

    assert calls == ["static", "probe"]
    assert set(result) == {"static", "probe"}
    for role, w in result.items():
        assert not w.is_deleted(), f"W[{role}] came back deleted"
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["static"])), np.eye(2)[None])
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["probe"])), 2.0 * np.eye(2)[None])


def test_a_single_role_scheme_never_spills(monkeypatch):
    """COHSEX (one request, the static role) must take zero host round
    trips -- ``_store_role`` never spills the LAST role, and with one
    request the static role IS the last one."""
    from gw import screening
    from gw.screening import ScreeningRequest

    spilled = []
    real_spill = collectives.spill_to_host

    def _counting_spill(arr):
        spilled.append(True)
        return real_spill(arr)

    def _fake_compute_static_w(wfns, V_q, quad, *, role, **kwargs):
        return jnp.eye(2, dtype=jnp.complex128)[None]

    monkeypatch.setattr(screening, "compute_static_w", _fake_compute_static_w)
    monkeypatch.setattr(screening, "_gate_w", lambda *a, **k: None)
    monkeypatch.setattr(collectives, "spill_to_host", _counting_spill)

    result = screening.compute_screening(
        wfns=None, V_q=None, requests=[ScreeningRequest(0.0 + 0.0j, "static")],
        quad=None, e_ref=0.0,
        sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
        config=_StubConfig(), meta=_StubMeta(), mesh_xy=None,
        print_fn=lambda *a, **k: None)

    assert spilled == []
    assert set(result) == {"static"}
    assert not result["static"].is_deleted()
