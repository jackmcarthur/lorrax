"""Network startup policy: active-step topology, portability and overrides."""
import subprocess
from types import SimpleNamespace

import pytest

from runtime import network_env as net


def environment(**updates):
    return dict(NERSC_HOST='perlmutter', JAX_PLATFORMS='cuda,cpu',
                SLURM_STEP_NUM_NODES='4', **updates)


def test_single_node_step_inside_four_node_allocation():
    env = environment()
    env.update(SLURM_STEP_NUM_NODES='1', SLURM_JOB_NUM_NODES='4',
               SLURM_NNODES='4', SLURM_NODELIST='nid[1-4]')
    result = net.plan_network_environment(env, is_file=lambda _: pytest.fail())
    assert result['nodes'] == 1 and result['applied'] == {}


def test_multi_node_request_preserves_tuning_override():
    env = environment(NCCL_CROSS_NIC='0')
    before = dict(env)
    result = net.plan_network_environment(env, is_file=lambda _: True)
    assert result['policy'] == 'Perlmutter OFI'
    assert result['applied']['NCCL_NET'] == 'AWS Libfabric'
    assert result['applied']['NCCL_NET_PLUGIN'].startswith('/')
    assert 'NCCL_CROSS_NIC' not in result['applied']
    assert env == before


@pytest.mark.parametrize('key,value', [('NCCL_NET', 'Socket'),
                                      ('NCCL_NET', ''),
                                      ('NCCL_NET_PLUGIN', 'none'),
                                      ('NCCL_NET_PLUGIN', '/other/plugin.so')])
def test_explicit_transport_wins_without_mixing_site_settings(key, value):
    result = net.plan_network_environment(environment(**{key: value}),
                                          is_file=lambda _: pytest.fail())
    assert result['policy'] == 'explicit transport'
    assert result['applied'] == {}


@pytest.mark.parametrize('site', ['', 'other-cluster'])
def test_unknown_site_is_unchanged(site):
    env = environment()
    env['NERSC_HOST'] = site
    assert net.plan_network_environment(env)['applied'] == {}


@pytest.mark.parametrize('platform,platforms', [('cpu', 'cuda,cpu'),
                                              ('gpu', 'cpu'), ('gpu', 'cpu,cuda'),
                                              ('gpu', 'rocm')])
def test_non_cuda_never_selects_ofi(platform, platforms):
    env = environment()
    env['JAX_PLATFORMS'] = platforms
    assert net.plan_network_environment(env, platform=platform)['applied'] == {}


def test_allocation_metadata_is_not_evidence_of_step_placement():
    env = environment()
    del env['SLURM_STEP_NUM_NODES']
    env.update(SLURM_JOB_NUM_NODES='4', SLURM_NODELIST='nid[1-4]')
    result = net.plan_network_environment(env)
    assert result['multi_node'] is None and result['applied'] == {}


def test_step_list_fallback(monkeypatch):
    def expand(argv, **kwargs):
        assert argv == ['scontrol', 'show', 'hostnames', 'nid[1-2]']
        return SimpleNamespace(stdout='nid1\nnid2\n')
    monkeypatch.setattr(net.subprocess, 'run', expand)
    assert net.launch_topology({'SLURM_STEP_NODELIST': 'nid[1-2]'})[:2] == (2, True)


def test_unavailable_launcher_keeps_unknown_topology(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('scontrol', 5)
    monkeypatch.setattr(net.subprocess, 'run', fail)
    assert net.launch_topology({'SLURM_STEP_NODELIST': 'nid[1-2]'})[:2] == (None, None)


@pytest.mark.parametrize('world,local,result', [(4,4,(1,False)),
                                              (7,3,(None,True)),
                                              (7,4,(None,True)),
                                              (2,4,(None,None))])
def test_openmpi_local_group_does_not_assume_uniform_rank_placement(world, local, result):
    env = {'OMPI_COMM_WORLD_SIZE': str(world), 'OMPI_COMM_WORLD_LOCAL_SIZE': str(local)}
    assert net.launch_topology(env)[:2] == result


def test_missing_site_plugin_has_actionable_error():
    with pytest.raises(RuntimeError, match='explicit NCCL_NET_PLUGIN/NCCL_NET'):
        net.plan_network_environment(environment(), is_file=lambda _: False)


def test_apply_once_and_preserve_record_of_automatic_selection(monkeypatch):
    monkeypatch.setattr(net, '_configuration', None)
    monkeypatch.setattr(net.os, 'environ', environment())
    monkeypatch.setattr(net.Path, 'is_file', lambda _: True)
    messages = []
    first = net.configure_gpu_network(say=messages.append)
    second = net.configure_gpu_network(say=messages.append)
    assert first == second == net.network_configuration()
    assert len(messages) == 1
    assert net.os.environ['NCCL_NET'] == 'AWS Libfabric'
