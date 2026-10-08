"""Sector route replays from production recipes (CPU, a 64-device host mesh in a subprocess).

The recipes are their bank receipts' counts (production accuracy, four held supports):
* Ni 20^3 bispinor P64 (runs/Ni/09_prod120_c1200_t600_20261002/bispinor/sc_wsupport_20261007T0300Z):
  641 parents, 32 line sites, 1.59 GB upstream. At its earlier 14 sites every sector is local at
  36 GB (sides 9920 / 14880 / 24800, as that leg printed); at 32 sites TT's local round prices
  63.9 GB, so 36 GB runs the staged face route in 11 rounds of 59; 72 GB stays q-local.
* CrI3 24x24 bispinor P64 (08_sectfast2_sigma_20261007): 61 parents, 16 sites, 72 GB, 7.44 GB
  upstream: one staged round of 61. At that leg's sides (TT 25856, CC 19264, CT 17408) every eigh
  stack takes route (c), the TT reduced stack (61 x 18432^2) at 68.5 of 72 GB; at the recipe's
  conservative sides TT's H'_vv and both CT stacks would go to the mesh, so the eigh routes are
  decided per round at the round's shapes.
* Fe 4^3 bispinor P4 (13 parents, 28 sites): q-local at 40 GB, staged rounds of 4 at 4 GB.
* CrI3 6x6 bispinor P4 (7 parents, 19 sites, 40 GB): q-local.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPLAY = r"""
import json, numpy as np, jax
from types import SimpleNamespace as NS
from jax.sharding import Mesh
from gw.shared_pole_recipe import CapacityLedger
from gw.shared_pole_sectors import sector_route
import gw.shared_pole_execution as ex
from gw.shared_pole_capacity import staged_sector_bytes, staged_cross_bytes
from gw.shared_pole_execution import face_ritz_carrier, face_reduction_bytes, stage_width, staged_eigh
ex._mesh_eigh = lambda mesh, n: NS(batched_route='scan', mesh=mesh)   # the mesh plan needs the native bundle
CASES = {  # name: (lines, imaginary, n, (CC logical, packed), (T logical, packed), nk, mesh side, nq, [(GB, upstream GB)])
    'ni14': (14, 4, 1194, (1194, 1216), (598, 640), 8000, 8, 641, [(36, 1.59)]),
    'ni': (32, 4, 1194, (1194, 1216), (598, 640), 8000, 8, 641, [(36, 1.59), (72, 1.59)]),
    'cri3_p64': (16, 2, 3300, (3300, 3328), (1650, 1664), 576, 8, 61, [(72, 7.44)]),
    'fe': (28, 4, 432, (432, 432), (144, 144), 64, 2, 13, [(40, 3.12), (4, 1.81)]),
    'cri3_6': (19, 3, 978, (978, 980), (450, 452), 36, 2, 7, [(40, 10.27)]),
}
GB = 1e9
out = {}


def ledger_for(mesh, meta, budget, up_gb):
    ledger = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=budget * 10**9)
    up = ledger.reserve('upstream:0', resident_bytes_per_rank=int(up_gb * 1e9), workspace_bytes_per_rank=0)['stage']
    ledger.live_stages = (up,)
    return ledger, up


def round_decisions(mesh, ledger, up, rows, sides, cross_side, infinity):
    # One staged round at the given sides: TT, then CC beside TT's held outputs, then CT beside
    # both sectors' models; each eigh's route and each stage's width, as staged_sector decides.
    R, P = 61, 64
    per_rank = lambda e: -(-16 * int(e) * R // P)
    got, live = {}, (up,)
    for row in (rows[1], rows[0]):
        name, packed = row['sector'], int(row['packed_extent'])
        side, iw, carrier = sides[name], infinity[name], face_ritz_carrier(mesh, row['pole_budget'])
        held = per_rank(2 * packed * (side - 2 * iw) + 5 * packed * iw)
        dw = per_rank(packed * (side - 2 * iw))
        stacks, boundaries = staged_sector_bytes(parents=R, ranks=P, side=side, carrier=carrier, packed=packed)
        program = lambda w: face_reduction_bytes(mesh, w, rows=packed, side=side, carrier=carrier, retain_span=True)
        stage = stage_width(ledger, stacks + held + dw, program, R, live=live)
        routes = []
        for m, bound in boundaries:
            plan = staged_eigh(mesh, (R, m, m), bound + held, ledger=ledger, live=live, label=name)
            row_ = ledger.preview(resident_bytes_per_rank=bound + held + -(-R // P) * 8 * m * m * 16,
                                  workspace_bytes_per_rank=0, concurrent_with=live)
            routes.append(('c' if plan.batched_route == 'batch_reshard' else 'mesh',
                           round(row_['aggregate_bytes_per_rank'] / GB, 2)))
        got[name] = dict(stage=stage, eighs=routes)
        outputs = ledger.reserve(f'held.{name}.{len(ledger.entries)}', resident_bytes_per_rank=held + per_rank(2 * packed * side + side * 2 * (carrier or side // 2)),
                                 workspace_bytes_per_rank=0, concurrent_with=live)['stage']
        live = (*live, outputs)
    models = ledger.reserve(f'kept.{len(ledger.entries)}', resident_bytes_per_rank=sum(per_rank(2 * int(r['packed_extent']) * sides[r['sector']]) for r in rows),
                            workspace_bytes_per_rank=0, concurrent_with=(up,))['stage']
    stacks, boundaries = staged_cross_bytes(parents=R, ranks=P, side=cross_side, rows=[r['packed_extent'] for r in rows])
    routes = []
    for m, bound in boundaries:
        plan = staged_eigh(mesh, (R, m, m), bound, ledger=ledger, live=(up, models), label='CT')
        routes.append('c' if plan.batched_route == 'batch_reshard' else 'mesh')
    got['CT'] = dict(eighs=routes)
    return got


for name, (lines, imag, n, cc, tt, nk, side, nq, budgets) in CASES.items():
    mesh = Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ('x', 'y'))
    z = [(0.19 + 0.09 * i, 0.1911) for i in range(lines)] + [(0, 0.12 * 3 ** j) for j in range(imag)]
    z += [(0.928, 0.1911), (2.131, 0.1911), (0, 0.201), (0, 1.714)]
    recipe = dict(accuracy='production', n=n, role=[0] * lines + [1] * imag + [3, 3, 4, 4],
                  held=[False] * (lines + imag) + [True] * 4, fit_ids=list(range(lines + imag)),
                  distinct_id=list(range(lines + imag + 4)), z_ry=[dict(real=a, imag=b) for a, b in z],
                  line_direction_cap=None, imaginary_width=None, infinity_width=None, pole_budget=None)
    for budget, up_gb in budgets:
        meta = NS(nk_tot=nk, nspinor=4, n_rmu=n, n_rmu_padded=cc[1], shared_pole_recipe=recipe)
        ledger, up = ledger_for(mesh, meta, budget, up_gb)
        meta.shared_pole_capacity = ledger
        bases = [NS(n_logical=cc[0], n_packed=cc[1]), NS(n_logical=tt[0], n_packed=tt[1])]
        mode, rows, route = sector_route(meta, NS(backend=NS(linalg='local')), bases, nq, mesh_xy=mesh, upstream=(up,))
        row = dict(mode=mode, route=route, modes=[r['mode'] for r in rows] + [rows[0]['joint']['mode']],
                   sides=[r['conservative_pencil_side'] for r in (*rows, rows[0]['joint'])],
                   tiles={k: v['selection'] for k, v in rows[0].get('staged_route', {}).items() if k != 'CT'})
        if name == 'cri3_p64':
            row['actual'] = round_decisions(mesh, ledger, up, rows, dict(TT=25856, CC=19264), 17408, dict(TT=640, CC=416))
            row['conservative'] = round_decisions(mesh, ledger, up, rows, dict(TT=32000, CC=20800), 29700,
                                                  dict(TT=640, CC=416))
        out[f'{name}/{budget}'] = row
print('REPLAY' + json.dumps(out))
"""


def test_sector_route_replay():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'src', *(root / 'services' / s / 'src' for s in
                             ('distrib_la', 'minimax', 'symmetry_maps', 'wfn_loader', 'lxkit'))]
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1',
               XLA_FLAGS='--xla_force_host_platform_device_count=64',
               PYTHONPATH=os.pathsep.join(map(str, paths)))
    run = subprocess.run([sys.executable, '-c', REPLAY], env=env, capture_output=True, text=True, timeout=900)
    line = [l for l in run.stdout.splitlines() if l.startswith('REPLAY')]
    assert line, run.stderr[-2000:]
    got = json.loads(line[0][len('REPLAY'):])
    assert got['ni14/36']['mode'] == 'local' and got['ni14/36']['sides'] == [9920, 14880, 24800]
    ni = got['ni/36']
    assert ni['modes'][:2] == ['local', 'face'] and ni['mode'] == 'face'
    assert ni['sides'] == [15680, 23520, 39200]
    assert ni['route'] == dict(width=59, rounds=11) and ni['tiles'] == dict(CC=52, TT=21)
    assert got['ni/72']['mode'] == 'local' and got['ni/72']['route'] == dict(width=64, rounds=11)
    cri3 = got['cri3_p64/72']
    assert cri3['mode'] == 'face' and cri3['route'] == dict(width=61, rounds=1)
    assert cri3['sides'][:2] == [20800, 32000] and cri3['tiles'] == dict(CC=9, TT=4)
    actual = cri3['actual']
    # Every stack keeps route (c) at the leg's sides; the TT reduced stack is the tightest.
    assert all(r == 'c' for name in ('TT', 'CC') for r, _ in actual[name]['eighs']), actual
    assert actual['CT']['eighs'] == ['c', 'c'], actual
    assert 68 < actual['TT']['eighs'][2][1] < 72, actual['TT']
    assert actual['TT']['stage'] >= 2 and actual['CC']['stage'] >= actual['TT']['stage'], actual
    # At the recipe's conservative sides TT's H'_vv and the CT stacks would leave route (c).
    conservative = cri3['conservative']
    assert conservative['TT']['eighs'][0][0] == 'mesh' and conservative['CT']['eighs'] == ['mesh', 'mesh']
    assert got['fe/40']['mode'] == 'local' and got['fe/40']['route'] == dict(width=4, rounds=4)
    assert got['fe/4']['mode'] == 'face' and got['fe/4']['route'] == dict(width=4, rounds=4)
    assert got['cri3_6/40']['mode'] == 'local' and got['cri3_6/40']['route'] == dict(width=4, rounds=2)
