"""Sector route replay of Ni 20^3 bispinor at P64 (641 parents, linalg local).

Ni's recipe from its map-0 bank receipt (runs/Ni/09_prod120_c1200_t600_20261002/
bispinor/sc_wsupport_20261007T0300Z): 32 line sites at Im z = 0.1911 Ry, four imaginary
supports, four held; CC 1194 x 1216 packed, TT 3 x 598 x 640 packed, 1.59 GB upstream.
At its earlier 14 sites every sector is local at 36 GB (sides 9920 / 14880 / 24800, as
that leg printed); at 32 sites TT's local reduction prices 63.9 GB and the constructor
goes to the face at 36 GB, local again at 72 GB. The decoupled route never takes 641
parents on 64 ranks (it ran out of memory there with lever 1, fdebbbd63).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from gw.shared_pole_execution import decoupled_route

REPLAY = r"""
import json, numpy as np, jax
from types import SimpleNamespace as NS
from jax.sharding import Mesh
from gw.shared_pole_recipe import CapacityLedger
from gw.shared_pole_sectors import sector_execution
mesh = Mesh(np.array(jax.devices()).reshape(8, 8), ('x', 'y'))
out = {}
for lines in (14, 32):
    z = [(0.19 + 0.09 * i, 0.1911) for i in range(lines)] + [(0, h) for h in (0.1176, 0.3435, 1.0031, 2.9297)]
    z += [(0.928, 0.1911), (2.131, 0.1911), (0, 0.201), (0, 1.714)]
    recipe = dict(accuracy='production', n=1194, role=[0] * lines + [1] * 4 + [3, 3, 4, 4],
                  held=[False] * (lines + 4) + [True] * 4, fit_ids=list(range(lines + 4)),
                  distinct_id=list(range(lines + 8)), z_ry=[dict(real=a, imag=b) for a, b in z],
                  line_direction_cap=75, imaginary_width=299, infinity_width=150, pole_budget=2150)
    for budget in (36, 72):
        meta = NS(nk_tot=8000, nspinor=4, n_rmu=1194, n_rmu_padded=1216, shared_pole_recipe=recipe)
        ledger = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=budget * 10**9)
        up = ledger.reserve('photon_endpoints_and_V:0', resident_bytes_per_rank=1585812480,
                            workspace_bytes_per_rank=0)['stage']
        ledger.live_stages = (up,)
        meta.shared_pole_capacity = ledger
        mode, rows = sector_execution(meta, NS(backend=NS(linalg='local')),
                                      [NS(n_logical=1194, n_packed=1216), NS(n_logical=598, n_packed=640)],
                                      641, mesh_xy=mesh, upstream=(up,))
        out[f'{lines}/{budget}'] = dict(resolved=mode, modes=[r['mode'] for r in rows] + [rows[0]['joint']['mode']],
                                        sides=[r['conservative_pencil_side'] for r in (*rows, rows[0]['joint'])])
print('REPLAY' + json.dumps(out))
"""


def test_decoupled_route_only_below_one_parent_per_rank():
    assert not decoupled_route(641, 1, 64)       # Ni 20^3 at P64: face rounds
    assert decoupled_route(61, 3, 64)            # CrI3 24x24 at P64
    assert not decoupled_route(61, 61, 64)       # one round holds every parent


def test_ni_route_replay():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'src', *(root / 'services' / s / 'src' for s in
                             ('distrib_la', 'minimax', 'symmetry_maps', 'wfn_loader', 'lxkit'))]
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1',
               XLA_FLAGS='--xla_force_host_platform_device_count=64',
               PYTHONPATH=os.pathsep.join(map(str, paths)))
    run = subprocess.run([sys.executable, '-c', REPLAY], env=env, capture_output=True, text=True, timeout=600)
    line = [l for l in run.stdout.splitlines() if l.startswith('REPLAY')]
    assert line, run.stderr[-2000:]
    got = json.loads(line[0][len('REPLAY'):])
    assert got['14/36'] == dict(resolved='local', modes=['local'] * 3, sides=[9920, 14880, 24800])
    assert got['32/36']['modes'][:2] == ['local', 'face'] and got['32/36']['resolved'] == 'face'
    assert got['32/36']['sides'] == [15680, 23520, 39200]
    assert got['32/72'] == dict(resolved='local', modes=['local'] * 3, sides=[15680, 23520, 39200])
    # The face rounds of 641 parents are warned with the q-local need, once (32 sites, 36 GB).
    warned = [l for l in run.stderr.splitlines() if 'RuntimeWarning: shared-pole sectors' in l]
    assert len(warned) == 1 and 'TT local needs 63.9 GB/rank' in warned[0], run.stderr[-2000:]
    assert 'memory_per_device_gb >= 64' in warned[0]
