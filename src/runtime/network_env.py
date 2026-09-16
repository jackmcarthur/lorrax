"""Select site network defaults before JAX/NCCL initialization.

Topology comes from the current launch, never the enclosing allocation.
No MPI or GPU library is imported here. Unknown sites/topologies preserve
NCCL's existing policy; site-specific network settings stay in this owner.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

_PERLMUTTER_PLUGIN = (
    '/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib/libnccl-net.so'
)
_PERLMUTTER_DEFAULTS = {
    'NCCL_NET': 'AWS Libfabric',
    'NCCL_NET_PLUGIN': _PERLMUTTER_PLUGIN,
    'NCCL_NET_GDR_LEVEL': 'PHB',
    'FI_CXI_DISABLE_HOST_REGISTER': '1',
    'NCCL_CROSS_NIC': '2',
    'NCCL_SOCKET_IFNAME': 'hsn',
}
_configuration = None


def _positive_int(value):
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def launch_topology(env):
    """Return (node count if known, multi-node if known, evidence source)."""
    count = _positive_int(env.get('SLURM_STEP_NUM_NODES'))
    if count is not None:
        return count, count > 1, 'Slurm step'
    nodelist = env.get('SLURM_STEP_NODELIST')
    if nodelist:
        try:
            hosts = subprocess.run(
                ['scontrol', 'show', 'hostnames', nodelist], check=True,
                capture_output=True, text=True, timeout=5,
            ).stdout.split()
            if hosts:
                count = len(set(hosts))
                return count, count > 1, 'Slurm step host list'
        except (OSError, subprocess.SubprocessError):
            pass
    # Do not divide world/local sizes: heterogeneous placement is valid.
    world = _positive_int(env.get('OMPI_COMM_WORLD_SIZE'))
    local = _positive_int(env.get('OMPI_COMM_WORLD_LOCAL_SIZE'))
    if world is not None and local is not None and local <= world:
        return (1, False, 'Open MPI local group') if local == world else (
            None, True, 'Open MPI local group')
    return None, None, 'unknown'


def plan_network_environment(env, *, platform='gpu', is_file=None):
    """Resolve defaults without modifying the caller's environment."""
    nodes, multiple, topology_source = launch_topology(env)
    result = dict(nodes=nodes, multi_node=multiple, topology_source=topology_source,
                  policy='unchanged', applied={})
    platforms = env.get('JAX_PLATFORMS', '').lower().split(',')
    if platform == 'cpu' or platforms[0] == 'cpu' or 'rocm' in platforms:
        result['policy'] = 'non-CUDA'
        return result
    if 'NCCL_NET' in env or 'NCCL_NET_PLUGIN' in env:
        result['policy'] = 'explicit transport'
        return result
    if multiple is not True or env.get('NERSC_HOST') != 'perlmutter':
        return result
    # SLURM_NETWORK=no_vni must reach srun before it creates the step. Mixed
    # MPI I/O and OFI/NCCL initialization failed without this launch setting;
    # changing os.environ here cannot repair an existing Slingshot VNI
    # allocation. Preserve the current transport unless the prerequisite is
    # present. See docs/environment/machines/perlmutter.md for the launch recipe.
    network_options = {value.strip() for value in
                       env.get('SLURM_NETWORK', '').split(',')}
    if 'no_vni' not in network_options:
        result['policy'] = 'unchanged (automatic OFI requires launch-time SLURM_NETWORK=no_vni)'
        return result
    # An absolute plugin name avoids changing LD_LIBRARY_PATH after Python
    # has started, which cannot reliably change the loader's search path.
    exists = is_file or (lambda path: Path(path).is_file())
    if not exists(_PERLMUTTER_PLUGIN):
        raise RuntimeError(
            'Multi-node Perlmutter CUDA startup cannot find the site OFI '
            f'plugin {_PERLMUTTER_PLUGIN}. Load the supported NCCL environment '
            'or provide an explicit NCCL_NET_PLUGIN/NCCL_NET configuration.')
    result['policy'] = 'Perlmutter OFI'
    result['applied'] = {
        name: value for name, value in _PERLMUTTER_DEFAULTS.items()
        if name not in env
    }
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
    say(f"[runtime] NCCL network: {placement}; {decision['policy']} "
        '(resolved before backend initialization).')
    return dict(decision)


def network_configuration():
    """Return the recorded request; NCCL's logs attest the loaded provider."""
    return dict(_configuration) if _configuration is not None else None
