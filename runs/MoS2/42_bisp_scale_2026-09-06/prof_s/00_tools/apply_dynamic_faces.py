"""Replace only the Sigma tau Green transport after the pinned controls finish."""
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path);args=parser.parse_args()
p=Path(__file__).resolve().parents[5]/'src/gw/ppm_tau_kernel.py'
s=p.read_text();a=s.index('    from distrib_la import gemm_plan',s.index('def _get_sigma_kij_kernel'));b=s.index('    def _bracketed_face',a)
t=s[a:b].replace('nq=k_unfold_plan.n_parent','nq=k_unfold_plan.n_full')
t=t.replace('    def _g_from_selector', '\n'.join([
    '    from common.shard_map import shard_map',
    '    specs = (P(None, None, "x", "y"), P(None, "x", None, "y"))',
    '',
    '    def unfold(xn, yr):',
    '        return (k_unfold_plan.unfold_face(',
    '            xn, vertex=0, spin_axis=1, mu_axis=2, mesh_axis="x"),',
    '                k_unfold_plan.unfold_face(',
    '            yr, vertex=0, spin_axis=2, mu_axis=3, mesh_axis="y"))',
    '    unfold = shard_map(unfold, mesh=mesh_xy, in_specs=specs,',
    '                       out_specs=specs, check_vma=False)',
    '',
    '    def _g_from_selector']))
t=t.replace('        if sel.dtype == jnp.bool_:', '\n'.join([
    '        xn, yr = unfold(xn, yr)',
    '        rows = jnp.asarray(k_unfold_plan.irr_idx)',
    '        sel = jnp.take(jnp.reshape(sel, E.shape), rows, axis=0)',
    '        E = jnp.take(E, rows, axis=0)',
    '        if sel.dtype == jnp.bool_:']))
t=t.replace(',\n                    k_unfold_plan=k_unfold_plan', '').replace(',\n                               k_unfold_plan=k_unfold_plan','').replace(',\n                k_unfold_plan=k_unfold_plan','').replace(',\n                           k_unfold_plan=k_unfold_plan','')
assert 'k_unfold_plan=k_unfold_plan' not in t
(args.output or p).write_text(s[:a]+t+s[b:])
