"""Sector route replays from production recipes (CPU, a 64-device host mesh in a subprocess).

The recipes are their bank receipts' counts (production accuracy, four held supports):
* Ni 20^3 bispinor P64 (runs/Ni/09_prod120_c1200_t600_20261002/bispinor/sc_wsupport_20261007T0300Z):
  641 parents, 32 line sites, 1.59 GB upstream. At its earlier 14 sites every sector is local at
  36 GB (sides 9920 / 14880 / 24800, as that leg printed); at 32 sites TT's local reduction prices
  63.9 GB, the decoupled stacks do not fit (it ran out of memory there with lever 1, fdebbbd63),
  and the face rounds hold 17 parents each (one before the whole-price step), warned; local at 72 GB.
* CrI3 24x24 bispinor P64 (04_sectfast2_20261006): 61 parents, 16 sites, 72 GB, 7.44 GB upstream:
  decoupled.
* Fe 4^3 bispinor P4 (runs/DEV/782_sectfast_20261006 n1/m3): 13 parents, 28 sites: decoupled at
  40 GB, face rounds at 4 GB (the decoupled leg n1 peaked at 5.24 GB there).
* CrI3 6x6 bispinor P4 (n2): 7 parents, 19 sites, 40 GB: decoupled.
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
from gw.shared_pole_sectors import sector_execution, decoupled_admission
CASES = {  # name: (lines, imaginary, n, (CC logical, packed), (T logical, packed), nk, mesh side, nq, [(GB, upstream GB)])
    'ni14': (14, 4, 1194, (1194, 1216), (598, 640), 8000, 8, 641, [(36, 1.59)]),
    'ni': (32, 4, 1194, (1194, 1216), (598, 640), 8000, 8, 641, [(36, 1.59), (72, 1.59)]),
    'cri3_p64': (16, 2, 3300, (3300, 3328), (1650, 1664), 576, 8, 61, [(72, 7.44)]),
    'fe': (28, 4, 432, (432, 432), (144, 144), 64, 2, 13, [(40, 3.12), (4, 1.81)]),
    'cri3_6': (19, 3, 978, (978, 980), (450, 452), 36, 2, 7, [(40, 10.27)]),
}
out = {}
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
        ledger = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=budget * 10**9)
        up = ledger.reserve('upstream:0', resident_bytes_per_rank=int(up_gb * 1e9), workspace_bytes_per_rank=0)['stage']
        ledger.live_stages = (up,)
        meta.shared_pole_capacity = ledger
        mode, rows = sector_execution(meta, NS(backend=NS(linalg='local')),
                                      [NS(n_logical=cc[0], n_packed=cc[1]), NS(n_logical=tt[0], n_packed=tt[1])],
                                      nq, mesh_xy=mesh, upstream=(up,))
        row = dict(resolved=mode, modes=[r['mode'] for r in rows] + [rows[0]['joint']['mode']],
                   sides=[r['conservative_pencil_side'] for r in (*rows, rows[0]['joint'])],
                   decoupled=decoupled_admission(rows, nq, mesh_xy=mesh, ledger=ledger, upstream=(up,))['admitted'])
        if name == 'ni' and mode == 'face':
            # The face batch from every parent (face_batch_width); the face eigh priced with the
            # local plan (ponytail: the scalapack plan needs the native bundle, absent on CPU CI).
            import gw.shared_pole_capacity as cap
            from gw.shared_pole_execution import sector_batch_width
            from gw.gw_config import linalg_resolution
            cap.constructor_eigenplan = lambda mesh_xy, side, execution, room=None: cap._local_eigenplan(mesh_xy, int(side))
            row['widths'] = [sector_batch_width(meta, linalg_resolution({'linalg': 'local'}), recipe, rows,
                                                mesh=mesh, ledger=ledger, nq=q)[0] for q in (641, 18)]
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
    assert got['ni14/36']['resolved'] == 'local' and got['ni14/36']['sides'] == [9920, 14880, 24800]
    ni = got['ni/36']
    assert ni['modes'][:2] == ['local', 'face'] and ni['resolved'] == 'face' and not ni['decoupled']
    assert ni['sides'] == [15680, 23520, 39200]
    assert ni['widths'] == [17, 17]          # 17 parents per round (35.1 GB of 36); one before the step fix
    assert got['ni/72']['resolved'] == 'local'
    assert got['cri3_p64/72']['resolved'] == 'face' and got['cri3_p64/72']['decoupled']
    assert got['cri3_p64/72']['sides'][:2] == [20800, 32000]
    assert got['fe/40']['decoupled'] and not got['fe/4']['decoupled']
    assert got['cri3_6/40']['decoupled']
    # Face rounds at nq >= P are warned with the q-local need, once each: Ni at 36 GB, Fe at 4 GB.
    warned = [l for l in run.stderr.splitlines() if 'RuntimeWarning: shared-pole sectors' in l]
    assert len(warned) == 2, run.stderr[-2000:]
    assert 'TT local needs 63.9 GB/rank' in warned[0] and 'memory_per_device_gb >= 64 runs' in warned[0]
    assert 'the 13 parents run in face rounds on 4 ranks' in warned[1]
