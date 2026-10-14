"""Launch topology and site network defaults, resolved before JAX/NCCL start.

Topology comes from the current launch, never the enclosing allocation. No MPI
or GPU library is imported here. A site profile, ``config/<site>/site.toml``,
states what the site provides and requires; this module decides when that
applies, what a caller's export overrides, and what refuses. Unknown sites and
topologies keep NCCL's own policy.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import tomllib

#: Variables naming the site, in precedence order; the value is a directory of
#: ``config/``.  ``NERSC_HOST`` is NERSC's own name for the machine.
_SITE_ENVS = ('LORRAX_SITE_PROFILE', 'NERSC_HOST')
_CONFIG = Path(__file__).resolve().parents[2] / 'config'
#: libfabric CXI provider settings for a Slingshot machine with no profile.
#: Without the zero rendezvous threshold CXI sends small messages eagerly, and
#: cross-node NCCL send/recv (all_to_all, collective_permute) deadlocks.
_CXI_DEFAULTS = {'FI_CXI_DISABLE_HOST_REGISTER': '1', 'FI_CXI_RDZV_THRESHOLD': '0'}
_configuration = None


def _positive_int(value):
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def expand_nodelist(expr):
    """Hosts of a Slurm host list, in order: ``nid[001000-001001,001010]``."""
    def expand(item):
        m = re.search(r'\[([^\]]*)\]', item)
        if not m:
            return [item]
        head, tails = item[:m.start()], expand(item[m.end():])
        hosts = []
        for part in m.group(1).split(','):
            lo, _, hi = part.partition('-')
            hosts += [f'{head}{i:0{len(lo)}d}{tail}'
                      for i in range(int(lo), int(hi or lo) + 1) for tail in tails]
        return hosts
    return [host for item in re.findall(r'[^,\[]+(?:\[[^\]]*\][^,\[]*)*', expr)
            for host in expand(item)]


def launch_topology(env):
    """Return (node count if known, multi-node if known, evidence source)."""
    count = _positive_int(env.get('SLURM_STEP_NUM_NODES'))
    if count is None and env.get('SLURM_STEP_NODELIST'):
        count = len(set(expand_nodelist(env['SLURM_STEP_NODELIST'])))
    if count is not None:
        return count, count > 1, 'Slurm step'
    # Do not divide world/local sizes: heterogeneous placement is valid.
    world = _positive_int(env.get('OMPI_COMM_WORLD_SIZE'))
    local = _positive_int(env.get('OMPI_COMM_WORLD_LOCAL_SIZE'))
    if world is not None and local is not None and local <= world:
        return (1, False, 'Open MPI local group') if local == world else (
            None, True, 'Open MPI local group')
    return None, None, 'unknown'


def site_profile(env=os.environ):
    """The selected ``config/<site>/site.toml`` as a dict, ``site`` its name; or None."""
    for var in _SITE_ENVS:
        name = env.get(var, '').strip()
        if not name:
            continue
        path = _CONFIG / name / 'site.toml'
        if path.is_file():
            with path.open('rb') as f:
                return {'site': name, **tomllib.load(f)}
        if var == 'LORRAX_SITE_PROFILE':
            raise RuntimeError(
                f'LORRAX_SITE_PROFILE={name!r} names no site profile: {path} does '
                f'not exist. Name a directory of {_CONFIG} that holds a site.toml, '
                'or unset it.')
    return None


def _sockets_refusal(site, nodes, cause, step_network):
    """Name why a profiled multi-node step would use sockets, and the relaunch."""
    placement = f'{nodes} nodes' if nodes is not None else 'multiple nodes'
    relaunch = (f'Relaunch with `SLURM_NETWORK={step_network} srun ...` (`lx run` '
                'exports it for every multi-node GPU step), and leave ' if step_network
                else 'Leave ')
    return (f'Refusing multi-node {site} CUDA startup on {placement}: {cause}, so '
            f'NCCL would run over TCP sockets. {relaunch}NCCL_NET unset or set to '
            'an OFI provider.')


def plan_network_environment(env, *, platform='gpu', is_file=None, cxi=None):
    """Resolve defaults without modifying the caller's environment."""
    nodes, multiple, topology_source = launch_topology(env)
    result = dict(nodes=nodes, multi_node=multiple, topology_source=topology_source,
                  policy='unchanged', applied={}, warnings=[])
    platforms = env.get('JAX_PLATFORMS', '').lower().split(',')
    if platform == 'cpu' or platforms[0] == 'cpu' or 'rocm' in platforms:
        result['policy'] = 'non-CUDA'
        return result
    profile = site_profile(env)
    nccl = (profile or {}).get('nccl') if multiple is True else None
    step_network = (nccl or {}).get('step_network')
    if 'NCCL_NET' in env or 'NCCL_NET_PLUGIN' in env:
        if nccl and env.get('NCCL_NET', '').strip().lower() == 'socket':
            raise RuntimeError(_sockets_refusal(
                profile['site'], nodes, 'NCCL_NET=Socket is set', step_network))
        result['policy'] = 'explicit transport'
        return result
    if multiple is not True:
        return result
    if nccl:
        # The option must reach srun before it creates the step; changing
        # os.environ here cannot repair the step's network allocation.
        options = {value.strip() for value in env.get('SLURM_NETWORK', '').split(',')}
        if step_network and step_network not in options:
            raise RuntimeError(_sockets_refusal(
                profile['site'], nodes, f'the step was created without SLURM_NETWORK='
                f"{step_network} (SLURM_NETWORK={env.get('SLURM_NETWORK')!r})",
                step_network))
        defaults = nccl.get('env', {})
        # An absolute plugin path, because LD_LIBRARY_PATH cannot change the
        # loader's search path once Python has started.
        plugin = defaults.get('NCCL_NET_PLUGIN')
        if plugin and not (is_file or os.path.isfile)(plugin):
            raise RuntimeError(
                f"Multi-node {profile['site']} CUDA startup cannot find the site OFI "
                f'plugin {plugin}. Load the supported NCCL environment or export '
                'NCCL_NET_PLUGIN/NCCL_NET.')
        result['policy'] = f"{profile['site']} profile"
    else:
        on_cxi = cxi if cxi is not None else os.path.isdir('/sys/class/cxi')
        defaults = _CXI_DEFAULTS if on_cxi else {}
        if on_cxi:
            result['policy'] = 'generic Slingshot'
            site = next((env[v] for v in _SITE_ENVS if env.get(v, '').strip()),
                        'this machine')
            result['warnings'].append(
                f'Slingshot (CXI) NICs but no site profile for {site} '
                f'({_CONFIG}/<site>/site.toml, selected by LORRAX_SITE_PROFILE or '
                f'NERSC_HOST); applied only the generic CXI settings '
                f'{sorted(_CXI_DEFAULTS)}.')
        path = env.get('LD_LIBRARY_PATH', '').split(':')
        if not any(d and os.path.isfile(os.path.join(d, 'libnccl-net.so')) for d in path):
            result['warnings'].append(
                'multi-node CUDA step with neither NCCL_NET nor a net plugin '
                '(NCCL_NET_PLUGIN, or libnccl-net.so on LD_LIBRARY_PATH): NCCL picks '
                'its own transport and may fall back to TCP sockets.')
    result['applied'] = {name: value for name, value in defaults.items() if name not in env}
    return result


def configure_gpu_network(*, platform='gpu', say=print):
    """Apply a process-lifetime startup decision once, before backend creation."""
    global _configuration
    if _configuration is not None:
        return dict(_configuration)
    decision = plan_network_environment(os.environ, platform=platform)
    if decision['policy'] == 'non-CUDA':
        return decision
    os.environ.update(decision['applied'])
    _configuration = decision
    nodes = decision['nodes']
    placement = (f'{nodes} node(s)' if nodes is not None else
                 'multiple nodes' if decision['multi_node'] else 'unknown placement')
    applied = f"; set {', '.join(decision['applied'])}" if decision['applied'] else ''
    say(f"[runtime] NCCL network: {placement}; {decision['policy']}{applied} "
        '(resolved before backend initialization).')
    for warning in decision['warnings']:
        say(f'[runtime] WARNING: {warning}')
    return dict(decision)


def network_configuration():
    """Return the recorded request; NCCL's logs attest the loaded provider."""
    return dict(_configuration) if _configuration is not None else None
