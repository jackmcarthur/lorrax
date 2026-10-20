"""Route completeness (owner, 2026-10-19): a stage whose candidates hold a whole N_mu-scale matrix
on a rank needs a distributed candidate; none is GATE nmu2_no_backup (lxkit.memory.require_backup).
A Gamma-only replica on a 4-device CPU mesh: one parent (nb = 1), N_mu = 64."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from jax.sharding import Mesh  # noqa: E402

from lxkit import memory  # noqa: E402

NMU = 64


@pytest.fixture
def gamma(monkeypatch):
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    memory.reset(run=True)
    memory.set_budget(1 << 30)
    memory.set_nmu(NMU)
    monkeypatch.setattr(memory, "NMU2_REPORT", False)
    yield Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    memory.reset(run=True)


def test_a_whole_nmu_route_with_no_mesh_backup_is_refused(gamma, monkeypatch):
    whole = memory.Price(xla=memory.Term(16 * NMU * NMU, source="one whole parent"), whole=NMU)
    with pytest.raises(RuntimeError, match="GATE nmu2_no_backup: got: gamma passivity"):
        memory.admit("gamma passivity", (("local", whole),), 1 << 30)
    # The census lists it instead.
    monkeypatch.setattr(memory, "NMU2_REPORT", True)
    memory.admit("gamma passivity", (("local", whole),), 1 << 30)
    assert any("nmu2_no_backup" in f and "gamma passivity" in f for f in memory.findings())


def test_a_whole_route_beside_a_mesh_route_or_a_small_matrix_passes(gamma):
    whole = memory.Price(xla=memory.Term(16 * NMU * NMU, source="one whole parent"), whole=NMU)
    face = memory.Price(xla=memory.Term(spread=16 * NMU * NMU, source="faces"), ranks=4)
    assert memory.admit("gamma line", (("local", whole), ("face", face)), 1 << 30) == "local"
    assert memory.admit("gamma line", (("local", whole), ("face", lambda: face)), 1 << 30) == "local"
    small = memory.Price(xla=memory.Term(16 * 8 * 8, source="a band tile"), whole=NMU // 2 - 1)
    assert memory.admit("gamma band tile", (("local", small),), 1 << 30) == "local"


def test_route_c_on_a_mesh_without_a_distributed_eigh_is_refused(gamma):
    import distrib_la
    plan = distrib_la.plan("eigh", gamma, n=NMU, backend="auto", budget_bytes=1 << 30)
    if plan.capacity_refusal is None:
        pytest.skip("this host resolves a distributed eigh")
    with pytest.raises(RuntimeError, match="GATE nmu2_no_backup: got: distrib_la eigh stack"):
        plan.stack_route((1, NMU, NMU), np.complex128)
