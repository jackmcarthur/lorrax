"""run30k plan census on a 16-device CPU mesh: every eigh plan the bank and the constructor
build, in the rooms the runtime helpers give them on the run's ledger, and every stack the
constructor's face programs decide while traced (jax.eval_shape on abstract operands)."""
import dataclasses, json, sys, traceback
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import lru_cache
from types import SimpleNamespace
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import lxkit, distrib_la
from distrib_la import loader, workspace
import importlib
_pl, _rs = (importlib.import_module("distrib_la." + m) for m in ("plan", "resolve"))
_pl.mesh_platform = _rs.mesh_platform = lambda mesh: "CUDA"
def query(ctx, n, cplx, device, host):      # cusolverDnXsyevd_bufferSize: 2x LAPACK (A100, measured)
    device._obj.value, host._obj.value = 2 * (16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n)), 0
    return 0
loader.get_lib = lambda platform: type("Lib", (), {"lrx_eigh_workspace_bytes": staticmethod(query)})()
lxkit.device_vendor = lambda device: "cuda"; workspace.device_vendor = lambda device: "cuda"
# The leg's capacity probe: cuSOLVERMp's whole-mesh eigh is present on the A100 leg.
_cap = lambda mesh_xy, *, n=None: ("cusolvermp", None)
_rs.resolve_eigh_capacity = _cap
if hasattr(_pl, "resolve_eigh_capacity"):
    _pl.resolve_eigh_capacity = _cap
from common import gpu_utils
gpu_utils._RUN_DEVICE_BUDGET_GB = 63.41
mesh = Mesh(np.array(jax.devices()[:16]).reshape(4, 4), ("x", "y"))
N, NLOG, BUDGET, NATIVE_EIGH = 30000, 29998, 63414637611, 7382397440
POLE_BUDGET, INFINITY_W, IMAG_W, LINE_CAP, STATE_W = 14999, 3750, 7500, 1875, 4612

# ---- the leg's vendor (cuSOLVERMp whole-mesh eigh), faked below the decision -------------
import gw.shared_pole_execution as spe
@lru_cache(maxsize=None)
def mesh_plan(mesh, n, backend):
    p = distrib_la.plan("eigh", mesh, n=int(n), backend="off", batched_route="auto", budget_bytes=0,
                        price_only=True)
    return dataclasses.replace(p, backend="cusolvermp")
spe._face_eigh = mesh_plan
def wbpr(plan, op, shapes, dtype):
    if plan.backend == "cusolvermp":
        return NATIVE_EIGH
    nb, n = int(shapes[0][0]), int(shapes[0][-1])
    return -(-nb // mesh.size) * 2 * (16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n))
distrib_la.workspace_bytes_per_rank = wbpr

census, refusals = [], []
_stack_route = _pl.Plan.stack_route
def stack_route(self, shape, dtype, op="eigh", traced=True):
    try:
        d = _stack_route(self, shape, dtype, op, traced)
    except Exception as e:
        refusals.append(dict(op=op, shape=list(map(int, shape)), error=str(e)[:300]))
        raise
    census.append(dict(op=op, shape=list(map(int, shape)), route=d.route,
                       price=None if d.program_bytes is None else int(d.program_bytes),
                       room=None if d.room is None else int(d.room),
                       exhausted=None if self.room_exhausted is None else self.room_exhausted.stage))
    return d
_pl.Plan.stack_route = stack_route
# Below the decision: the vendor solve as shapes (route_for -> stack_route already ran).
_batched = _pl.Plan.batched
def batched(self, A, *args, **kw):
    if self.backend == "cusolvermp" and self.op == "eigh":
        self.route_for(A.shape, A.dtype, isinstance(A, jax.core.Tracer))
        kw.pop("cut", None)
        w, v = jnp.linalg.eigh(A)
        return w, v
    return _batched(self, A, *args, **kw)
_pl.Plan.batched = batched

def stage(name, fn):
    try:
        out = fn()
        print(f"STAGE {name}: ok {out if out is not None else ''}", flush=True)
    except Exception as e:
        refusals.append(dict(stage=name, error=f"{type(e).__name__}: {str(e)[:400]}"))
        print(f"STAGE {name}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc(limit=6)

from gw.shared_pole_recipe import CapacityLedger
from gw.shared_pole_execution import (line_selection_execution, line_selection_room, face_batch_width,
                                      face_reduction_bytes, face_ritz_carrier, face_parent_program,
                                      face_eigh)
from gw.shared_pole_capacity import (face_eigh_room, check_eigh_room, ConstructorCapacity,
                                     constructor_eigenplan)
from gw.shared_pole_directions import charge_line_selection, port_extent
ledger = CapacityLedger(SimpleNamespace(nk_tot=1, nspinor=2, n_rmu=NLOG), mesh_xy=mesh,
                        device_budget_bytes=BUDGET)
face_bytes = 16 * N * N // mesh.size
ledger.reserve("coulomb_roots:2", resident_bytes_per_rank=face_bytes, workspace_bytes_per_rank=0)
ledger.live_stages = ("coulomb_roots:2",)
info = {}

# ---- bank ---------------------------------------------------------------------------------
def bank_line():
    carry = 2 * face_bytes
    execution, resident, ws = line_selection_execution([N], mesh=mesh, ledger=ledger, nq=1, carry=carry,
                                                       eigh_backend="auto")
    room = line_selection_room(ledger, resident, carry=carry)
    meta = SimpleNamespace(mu_basis=SimpleNamespace(active_mask=np.arange(N) < NLOG, n_packed=N),
                           n_rmu=NLOG, shared_pole_recipe={})
    sel = charge_line_selection(meta, mesh_xy=mesh, ordered=True, execution=execution, nq=1,
                                eigh_backend="auto", room=room)
    sel.plans["charge"].stack_route((1, N, N), np.complex128, "normal_eigh")   # polar extract
    info["line"] = dict(execution=execution, face_aggregate=resident + ws + carry + face_bytes, room=int(room))
    return info["line"]
stage("bank line selection", bank_line)

def bank_coulomb():
    from gw import response_bank as rb
    room = gpu_utils.shared_device_room("shared-pole bank Coulomb root eigh")
    p = distrib_la.plan("eigh", mesh, n=N, budget_bytes=room)
    p = dataclasses.replace(p, backend="cusolvermp") if p.backend != "cusolvermp" else p
    p.stack_route((1, N, N), np.complex128)
    return dict(room=int(room))
stage("bank Coulomb root eigh", bank_coulomb)

def isdf_charge():
    room = gpu_utils.shared_device_room("isdf zeta charge eigh")
    p = distrib_la.plan("eigh", mesh, n=N, budget_bytes=room)
    p = dataclasses.replace(p, backend="cusolvermp") if p.backend != "cusolvermp" else p
    p.stack_route((1, N, N), np.complex128)
    return dict(room=int(room))
stage("isdf zeta charge eigh", isdf_charge)

# ---- constructor (shared_pole_constructor face branch) -----------------------------------
meta = SimpleNamespace(n_rmu_padded=N)
resolution = SimpleNamespace(eigh_backend="auto", layout="local")
extent = port_extent(mesh)
carrier = face_ritz_carrier(mesh, POLE_BUDGET)
CONSERVATIVE, SIDE = 84480, 48640
upstream = ledger.live_stages
budget = None
def constructor_admission():
    global budget, face_room
    width, receipt = face_batch_width(
        meta, resolution, mesh=mesh, ledger=ledger, upstream=upstream, side=CONSERVATIVE, nq=1,
        selection=dict(sample_batch=4, selection_faces=None),
        program_bytes=lambda w: face_reduction_bytes(mesh, w, rows=N, side=CONSERVATIVE, carrier=carrier))
    budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh, ledger=ledger, upstream=upstream,
                                 execution="face")
    budget.ritz_budget = None
    budget.program_bytes = receipt["program_bytes_per_rank"]
    face_room = face_eigh_room(receipt["selection"], 0, stage="shared-pole constructor selection")
    budget.face_room = face_room
    info["constructor"] = dict(width=width, program_gb=receipt["program_bytes_per_rank"] / 1e9,
                               selection_gb=receipt["selection"]["aggregate_bytes_per_rank"] / 1e9,
                               reduction_gb=receipt["reduction"]["aggregate_bytes_per_rank"] / 1e9,
                               face_room=int(face_room))
    info["receipt"] = receipt
    return info["constructor"]
stage("constructor batch admission", constructor_admission)

def constructor_selection():
    eig = budget.eigenplan(N)
    for nb, op in ((1, "eigh"), (2, "eigh"), (1, "normal_eigh")):
        eig.stack_route((nb, N, N), np.complex128, op)
stage("constructor direction selection", constructor_selection)

def constructor_reduction():
    receipt = info["receipt"]
    admitted = budget.program_bytes
    actual = face_reduction_bytes(mesh, 1, rows=N, side=SIDE, carrier=carrier)
    budget.program_bytes = actual
    try:
        reserved = budget.plan(SIDE, phase="reduction")
    finally:
        budget.program_bytes = admitted
    row = receipt["reduction"]
    room = lambda price: face_eigh_room(dict(row, aggregate_bytes_per_rank=(
        row["aggregate_bytes_per_rank"] - budget.program_bytes + price)), 0,
        stage="shared-pole constructor reduction round")
    r = room(actual)
    info["reduction"] = dict(reserved_gb=reserved["aggregate_bytes_per_rank"] / 1e9,
                             price_gb=actual / 1e9, room=str(r))
    program = face_parent_program(mesh, True, True, POLE_BUDGET, False, SIDE, None,
                                  face_eigh(mesh, SIDE, room=r, backend="auto"), int(carrier))
    face = NamedSharding(mesh, P(None, "x", "y"))
    from gw.shared_pole_local import round_tables
    # The round's tables from the recipe's panel widths (4 fit samples x 2 orientations).
    states = 8
    # 8 states of 4612 retained columns (padded 5120): the run's pencil side 48640.
    widths = [extent(STATE_W)] * 8
    tables = round_tables(np.array([[STATE_W] * 8]), widths,
                          [0.19j, 0.62 + 0.19j, 0.21j, 2.02j, -0.19j, -0.62 - 0.19j, -0.21j, -2.02j],
                          [INFINITY_W], extent(INFINITY_W), column_extent=extent, ordered=True,
                          odd_moments=True)
    from gw.shared_pole_pencil import interleave_tables
    tables = interleave_tables(tables, int(mesh.shape["y"]))
    F, side = int(tables["order"].shape[-1]), int(tables["active"].shape[-1])
    assert side == SIDE, side
    S = lambda *shape: jax.ShapeDtypeStruct(shape, jnp.complex128, sharding=face)
    args = (tables["points"], tables["active"], S(1, N, F), S(1, N, F), S(1, N, F),
            tuple(S(1, N, extent(INFINITY_W)) for _ in range(5)))
    info["reduction"]["side"] = side
    try:
        program(*args)
    except _Traced:
        pass
    return info["reduction"]

class _Traced(Exception):
    pass
from distrib_la import _result_check as _rc
def fake_checked_program(fn, mesh, out_shardings, in_shardings=None, compiler_options=None):
    def call(*args):
        jax.eval_shape(_rc._in_phase(fn, "all"), *args)      # the whole-chain retry's decisions
        raise _Traced()
    return call
distrib_la.checked_program = fake_checked_program
spe.distrib_la = distrib_la
stage("constructor gram_reduction (eval_shape of face_parent_program)", constructor_reduction)

def constructor_passivity():
    # shared_pole_constructor: model_row = budget.plan(side, phase="model", sample_batch=len(held_ids))
    budget.batch_width = 1
    model_row = budget.plan(SIDE, phase="model", sample_batch=2)
    price = spe.face_check_bytes(mesh, N, extent(POLE_BUDGET))
    info["passivity"] = dict(model_gb=model_row["aggregate_bytes_per_rank"] / 1e9, price_gb=price / 1e9)
    r = check_eigh_room(model_row, price)
    plan = face_eigh(mesh, N, room=r, backend="auto")
    for nb in (1, 2):
        plan.stack_route((nb, N, N), np.complex128, "eigh")
    info["passivity"]["room"] = str(r)
    return info["passivity"]
stage("constructor passivity_held", constructor_passivity)

# ---- a missing room is refused where the plan is built, never at a stack ------------------
def missing_rooms():
    out = {}
    for label, call in (("room=0", lambda: face_eigh(mesh, SIDE, room=0, backend="auto")),
                        ("room=None", lambda: face_eigh(mesh, SIDE, room=None, backend="auto")),
                        ("no room", lambda: constructor_eigenplan(mesh, N, "face", backend="auto"))):
        try:
            call()
            out[label] = "built"
        except (RuntimeError, ValueError, TypeError) as e:
            out[label] = f"{type(e).__name__}: {str(e)[:900]}"
    info["missing_rooms"] = out
    return out
stage("missing rooms", missing_rooms)

print("RESULT " + json.dumps(dict(info={k: v for k, v in info.items() if k != "receipt"},
                                  census=census, refusals=refusals), default=str))
