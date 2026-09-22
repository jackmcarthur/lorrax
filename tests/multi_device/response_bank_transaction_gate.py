"""Collective frequency transactions: read-after-write and interrupted publication refusal."""
from runtime import initialize_communicator_stack, finalize_process
stack = initialize_communicator_stack(platform="gpu")
from pathlib import Path
import sys
import time
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_shared_pole_store import _fixture
from test_shared_pole_bank import _matrix
from file_io.shared_pole_store import (
    initialize_shared_pole_bank, validate_shared_pole_bank,
    shared_pole_bank_writer, read_shared_pole_bank, write_shared_pole_bank,
)

mesh = stack.mesh
meta, tables, recipe, identity = _fixture(mesh)
tables["sym"].trs_allowed = False
from gw.shared_pole_recipe import CapacityLedger
meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=2**20)
meta.shared_pole_capacity.reserve("fixture_live_bound", resident_bytes_per_rank=4096, workspace_bytes_per_rank=0)
meta.shared_pole_capacity.live_stages = ("fixture_live_bound",)
recipe.update(role_codes={"line":0,"imaginary":1,"infinity":2,"held_line":3,"held_imaginary":4},
    z_ry=np.array([.1j,.1j,.2+.1j]), role=np.array([0,1,3],np.int8),
    distinct_id=np.array([0,0,1]), held=np.array([False,False,True]),
    support_pair=np.array([[-1,-1],[-1,-1],[0,1]]), fit_ids=np.array([0]), held_ids=np.array([1]))
path = Path(sys.argv[1])
header = initialize_shared_pole_bank(path, meta=meta, tables=tables,
    recipe=recipe, identity=identity, mesh_xy=mesh)
args = dict(meta=meta, expected_identity=identity, mesh_xy=mesh)
w = _matrix(meta,mesh,samples=True,value=3)
m = _matrix(meta,mesh,samples=False,value=11)
try:
    with shared_pole_bank_writer(path, **args) as (io, header, write):
        write(q_span=(0,1), sample_span=(0,1), Wc=w)
        raise RuntimeError("interrupt before frequency commit")
except RuntimeError as error:
    assert "interrupt before frequency commit" in str(error)
try:
    validate_shared_pole_bank(path,expected_identity=identity,mesh_xy=mesh)
except ValueError as error:
    assert "not globally committed" in str(error)
else:
    raise AssertionError("interrupted bank was accepted")
path = path.with_name(path.stem + "_complete.h5")
header = initialize_shared_pole_bank(path, meta=meta, tables=tables,
    recipe=recipe, identity=identity, mesh_xy=mesh)
started=time.monotonic()
for sample in range(2):
    with shared_pole_bank_writer(path, **args) as (io, header, write):
        for q in range(3):
            write(q_span=(q,q+1),sample_span=(sample,sample+1),Wc=w,Wc_mirror=2*w)
        for q in range(3):
            saved=read_shared_pole_bank(io,(q,q+1),meta=meta,header=header,
                sample_span=(sample,sample+1),fields=("Wc","Wc_mirror"))
            assert bool(jnp.all(saved["Wc"] == w))
            assert bool(jnp.all(saved["Wc_mirror"] == 2*w))
            write(q_span=(q,q+1),sample_span=(sample,sample+1),dWc_ds=3*w,dWc_mirror_ds=4*w)
    actual=validate_shared_pole_bank(path,expected_identity=identity,mesh_xy=mesh)
    assert np.asarray(actual["sample_written"])[:,sample].all()
for q in range(3):
    write_shared_pole_bank(path,q_span=(q,q+1),M0=m,M1=m,M2=m,M3=m,**args)
actual=validate_shared_pole_bank(path,expected_identity=identity,mesh_xy=mesh,require_complete=True)
assert actual["complete"] and actual["final_commit"]
if stack.process_index == 0:
    print(f"PASS frequency transaction, literal mirrors, read-after-write, interrupted publication refusal, final commit: {time.monotonic()-started:.3f}s",flush=True)
finalize_process()
