"""run30k plan census on a 16-device CPU mesh: every eigh plan the bank and the constructor
build, in the rooms the runtime helpers give them on the run's ledger, and every stack the
constructor's face programs decide while traced (jax.eval_shape on abstract operands).

``LORRAX_REPLAY_DECK=deck.json`` replaces the run30k extents (keys of ``DECK``) with another
scalar ordered deck's; ``LORRAX_REPLAY_MEASURED=cal.csv`` (the calibration CSV: deck, stage,
outside, priced, measured in GB/rank) prints each of its ``deck_name`` rows beside this tree's
price for that production section (``memory.receipts``), and names the C4 -> C6 change of the
stages it does not replay."""
import dataclasses, json, os, sys, traceback
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import lru_cache
from types import SimpleNamespace
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import lxkit, distrib_la
from distrib_la import workspace
import importlib
_pl, _rs = (importlib.import_module("distrib_la." + m) for m in ("plan", "resolve"))
_pl.mesh_platform = _rs.mesh_platform = workspace.mesh_platform = lambda mesh: "CUDA"
lxkit.device_vendor = lambda device: "cuda"; workspace.device_vendor = lambda device: "cuda"
# The leg's capacity probe: cuSOLVERMp's whole-mesh eigh is present on the A100 leg.
_cap = lambda mesh_xy, *, n=None: ("cusolvermp", None)
_rs.resolve_eigh_capacity = _cap
if hasattr(_pl, "resolve_eigh_capacity"):
    _pl.resolve_eigh_capacity = _cap
from common import gpu_utils
from lxkit import memory
DECK = dict(deck_name="run30k", mesh_side=4, N=30000, NLOG=29998, BUDGET=63414637611,
            LIMIT=77.89e9, NATIVE_EIGH=7382397440, POLE_BUDGET=14999, INFINITY_W=3750,
            STATE_W=4612, BASELINE=3_130_000_000, CONSERVATIVE=84480, SIDE=48640)
if os.environ.get("LORRAX_REPLAY_DECK"):
    DECK.update(json.load(open(os.environ["LORRAX_REPLAY_DECK"])))
memory.set_budget(DECK["BUDGET"], limit_bytes=DECK["LIMIT"], per_node=4,
                  platform="NVIDIA A100-SXM4-80GB", linalg="distributed")
mesh = Mesh(np.array(jax.devices()[:DECK["mesh_side"] ** 2]).reshape(DECK["mesh_side"], -1),
            ("x", "y"))
N, NLOG, BUDGET, NATIVE_EIGH = (DECK[k] for k in ("N", "NLOG", "BUDGET", "NATIVE_EIGH"))
POLE_BUDGET, INFINITY_W, STATE_W = DECK["POLE_BUDGET"], DECK["INFINITY_W"], DECK["STATE_W"]

# ---- the leg's vendor (cuSOLVERMp whole-mesh eigh), faked below the decision -------------
import gw.shared_pole_execution as spe
@lru_cache(maxsize=None)
def mesh_plan(mesh, n, backend):
    p = distrib_la.plan("eigh", mesh, n=int(n), backend="off", batched_route="auto", budget_bytes=0,
                        price_only=True)
    return as_mp(p)
def as_mp(p):
    """A plan as the leg's cuSOLVERMp plan: its backend and its face operand layouts."""
    return dataclasses.replace(p, backend="cusolvermp", in_sharding=NamedSharding(p.mesh, P("x", "y")),
                               batch_in_sharding=NamedSharding(p.mesh, P(None, "x", "y")))
spe._face_eigh = mesh_plan
# cuSOLVERMp's workspace is a native query (GPU only): the run's measured 7.38 GB/rank. Local
# plans price theirs from the leg's measured cusolverDn row (distrib_la.workspace).
_wbpr = distrib_la.workspace_bytes_per_rank
distrib_la.workspace_bytes_per_rank = lambda plan, op, shapes, dtype: (
    NATIVE_EIGH if plan.backend == "cusolvermp" else _wbpr(plan, op, shapes, dtype))

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
# Below the decision: the library call as shapes, so the service's own checked chain compiles
# around it.  cuSOLVERMp holds an operand copy (it destroys an aliased one) and returns the
# vectors at A's face layout and the values replicated; its workspace is the query above.
import distrib_la._cusolvermp as _mp
def _mp_eigh(A, *, mesh, compute_evecs=True, block=None, side=None):
    face = NamedSharding(mesh, P("x", "y"))
    work = jax.lax.with_sharding_constraint(A * (1 + 0j), face)
    return (jax.lax.with_sharding_constraint(jnp.zeros(A.shape[-1], jnp.float64), NamedSharding(mesh, P())),
            jax.lax.with_sharding_constraint(work * 1, face))
_mp.distributed_eigh = _mp_eigh

def stage(name, fn):
    if name.startswith("constructor"):
        ledger.at_phase()                             # shared_pole_constructor's phase entry
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
from gw.shared_pole_local import book_check_eighs
from common import timing
# The driver's arrays live beside the ledger: leg 59998942 measured a 3.13 GB/rank floor
# ('spole other'; Sigma rule plan 3.17, QP 3.19), what the bind's baseline measures.
BASELINE = DECK["BASELINE"]
ledger = CapacityLedger(SimpleNamespace(nk_tot=1, nspinor=2, n_rmu=NLOG), mesh_xy=mesh,
                        device_budget_bytes=BUDGET, baseline_bytes=BASELINE)
# The room helpers that measure live arrays see the same floor.
memory.room = lambda stage="", *, bucketed=True: (
    memory.bucket if bucketed else int)(BUDGET - BASELINE)
ledger.room_bytes = BUDGET - BASELINE              # resolve_shared_pole_recipe's bind
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
    row = ledger.reserve("line_selection:0", resident_bytes_per_rank=resident,
                         workspace_bytes_per_rank=ws, concurrent_with=ledger.live_stages)
    memory.book(ledger.price(row), stage="bank.line_select")
    with timing.section("bank.line_select"):
        pass
    info["line"] = dict(execution=execution, face_aggregate=resident + ws + carry + face_bytes, room=int(room))
    return info["line"]
stage("bank line selection", bank_line)

def bank_coulomb():
    from gw import response_bank as rb
    room = gpu_utils.shared_device_room("shared-pole bank Coulomb root eigh")
    p = distrib_la.plan("eigh", mesh, n=N, budget_bytes=room)
    p = as_mp(p) if p.backend != "cusolvermp" else p
    p.stack_route((1, N, N), np.complex128)
    return dict(room=int(room))
stage("bank Coulomb root eigh", bank_coulomb)

def isdf_charge():
    room = gpu_utils.shared_device_room("isdf zeta charge eigh")
    p = distrib_la.plan("eigh", mesh, n=N, budget_bytes=room)
    p = as_mp(p) if p.backend != "cusolvermp" else p
    p.stack_route((1, N, N), np.complex128)
    return dict(room=int(room))
stage("isdf zeta charge eigh", isdf_charge)

# ---- constructor (shared_pole_constructor face branch) -----------------------------------
meta = SimpleNamespace(n_rmu_padded=N)
resolution = SimpleNamespace(eigh_backend="auto", layout="local")
extent = port_extent(mesh)
carrier = face_ritz_carrier(mesh, POLE_BUDGET)
CONSERVATIVE, SIDE = DECK["CONSERVATIVE"], DECK["SIDE"]
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
    with timing.section("spole.batch_admission"):
        budget.batch_width = width
        budget.plan(CONSERVATIVE, phase="selection", sample_batch=4)
        budget.live(())
    return info["constructor"]
stage("constructor batch admission", constructor_admission)

def constructor_selection():
    eig = budget.eigenplan(N)
    for nb, op in ((1, "eigh"), (2, "eigh"), (1, "normal_eigh")):
        eig.stack_route((nb, N, N), np.complex128, op)
stage("constructor direction selection", constructor_selection)

def constructor_reduction():
    # shared_pole_constructor: the reservation holds the round's operands (program 0); the
    # round's program is priced by its compiled figure in gram_reduction (price_round).
    receipt = info["receipt"]
    admitted = budget.program_bytes
    actual = face_reduction_bytes(mesh, 1, rows=N, side=SIDE, carrier=carrier)
    budget.program_bytes = 0
    try:
        with timing.section("spole.reduction_admission"):
            reserved = budget.plan(SIDE, phase="reduction", runs_in="spole.gram_reduction")
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
    with timing.section("spole.gram_reduction"):
        executable, _ = program.checked(*args).compiled(*args)
        spe.price_round(executable, mesh, budget.ledger.price(reserved))
    info["reduction"]["compiled_gb"] = executable.memory_analysis().temp_size_in_bytes / 1e9
    return info["reduction"]

from distrib_la import _result_check as _rc
_checked_program = distrib_la.checked_program
def census_checked_program(fn, mesh, out_shardings, in_shardings=None, compiler_options=None):
    """The real checked program (compiled on CPU), whose whole-chain retry decisions are also
    traced for the census (jax.eval_shape)."""
    call = _checked_program(fn, mesh, out_shardings, in_shardings, compiler_options)
    compiled = call.compiled
    def traced(*args):
        jax.eval_shape(_rc._in_phase(fn, "all"), *args)
        return compiled(*args)
    call.compiled = traced
    return call
distrib_la.checked_program = census_checked_program
spe.distrib_la = distrib_la
stage("constructor gram_reduction (compiled face_parent_program)", constructor_reduction)

def constructor_passivity():
    # shared_pole_constructor: model_row = budget.plan(side, phase="model", sample_batch=len(held_ids))
    budget.batch_width = 1
    model_row = budget.plan(SIDE, phase="model", sample_batch=2)
    price = spe.face_check_bytes(mesh, N, extent(POLE_BUDGET))
    info["passivity"] = dict(model_gb=model_row["aggregate_bytes_per_rank"] / 1e9, price_gb=price / 1e9)
    r = check_eigh_room(model_row, price)
    with timing.section("spole.passivity_held"):      # shared_pole_local.check_round
        book_check_eighs(face_eigh(mesh, N, room=r, backend="auto"), N)
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
print("SIZING " + memory.sizing()["line"])
print("CENSUS " + json.dumps(memory.findings()))

# ---- this tree's price per production section beside a measured leg ----------------------
#: What changed C4 -> C6 in the stages this replay does not rebuild.
DELTA = (("tau.memcheck", "unpriced (A5: provisional bound deleted)"),
         ("tau.setup", "unpriced (A5: provisional bound deleted)"),
         ("tau.window_arguments", "unpriced (A5: provisional bound deleted)"),
         ("spole.bank", "C4 less its outside term (A6: stage_outside deleted)"),
         ("spole.reduction_admission", "booked in spole.gram_reduction (runs_in), as in C4"),
         ("spole.", "ledger row: absolute (baseline + live rows + own); C4 an increment"),
         ("bank.line_", "ledger row: absolute (baseline + live rows + own); C4 an increment"),
         ("gw_jax.V_q", "tile admission host-only (A4); device price = booked parts"),
         ("", "formula unchanged C4 -> C6"))
if os.environ.get("LORRAX_REPLAY_MEASURED"):
    import csv
    priced = {r["stage"]: r["priced"] / 1e9 for r in memory.receipts()}
    rows = [r for r in csv.DictReader(open(os.environ["LORRAX_REPLAY_MEASURED"]))
            if r["deck"] == DECK["deck_name"]]
    print(f"PRICES {DECK['deck_name']}: stage | C4 priced | measured | C6 priced | gamma C6 | C6 basis")
    for r in sorted(rows, key=lambda r: r["stage"]):
        c4, meas = float(r["priced"]), float(r["measured"])
        if r["stage"] in priced:
            c6, basis = priced[r["stage"]], "replayed"
        else:
            basis = next(t for k, t in DELTA if r["stage"].startswith(k))
            c6 = (None if basis.startswith("unpriced") else
                  c4 - float(r["outside"] or 0) if r["stage"] == "spole.bank" else
                  None if basis.startswith(("ledger", "tile")) else c4)
        cell = "-" if c6 is None else f"{c6:.2f}"
        gamma = "-" if not c6 else f"{meas / c6:.2f}"
        print(f"PRICE {r['stage']} | {c4:.2f} | {meas:.2f} | {cell} | {gamma} | {basis}")
    for stage in sorted(set(priced) - {r["stage"] for r in rows}):
        print(f"PRICE {stage} | - | - | {priced[stage]:.2f} | - | replayed, not in the CSV")
