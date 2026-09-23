"""Network startup policy: active-step topology, portability and overrides."""
import subprocess
from types import SimpleNamespace

import pytest

from runtime import network_env as net


def environment(**updates):
    return dict(NERSC_HOST='perlmutter', JAX_PLATFORMS='cuda,cpu',
                SLURM_STEP_NUM_NODES='4', SLURM_NETWORK='no_vni', **updates)


@pytest.mark.parametrize('setting', [None, '', 'depth=64', 'not_no_vni'])
def test_uncertified_launch_does_not_enable_ofi(setting):
    env = environment()
    if setting is None:
        del env['SLURM_NETWORK']
    else:
        env['SLURM_NETWORK'] = setting
    result = net.plan_network_environment(env, is_file=lambda _: pytest.fail())
    assert result['applied'] == {}
    assert 'requires launch-time' in result['policy']


def test_no_vni_token_can_accompany_other_network_options():
    env = environment()
    env['SLURM_NETWORK'] = 'depth=64, no_vni'
    assert net.plan_network_environment(env, is_file=lambda _: True)['policy'] == 'Perlmutter OFI'


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


def test_ofi_selection_disables_cxi_eager_messages():
    # Without it every P16 no_vni sc.eigh leg deadlocked in cross-node NCCL
    # send/recv (runs/runtime/nccl_ofi_hang_20260923); an override still wins.
    applied = net.plan_network_environment(environment(), is_file=lambda _: True)['applied']
    assert applied['FI_CXI_RDZV_THRESHOLD'] == '0'
    kept = net.plan_network_environment(environment(FI_CXI_RDZV_THRESHOLD='16384'),
                                        is_file=lambda _: True)['applied']
    assert 'FI_CXI_RDZV_THRESHOLD' not in kept


_SITE_MODULE = '/opt/nersc/pe/modulefiles/nccl/2.29.2-cu13.lua'


@pytest.mark.skipif(not __import__('os').path.isfile(_SITE_MODULE),
                    reason='the NERSC nccl modulefile exists only on Perlmutter')
def test_defaults_carry_every_network_setting_of_the_site_module():
    import re
    text = open(_SITE_MODULE).read()
    settings = dict(re.findall(r'setenv\("([A-Z0-9_]+)",\s*"?([^")]*)"?\)', text))
    network = {k: v for k, v in settings.items()
               if k.startswith(('FI_', 'NCCL_')) and k not in ('NCCL_DIR', 'NCCL_HOME', 'NCCL_VERSION')}
    assert network, 'no network settings parsed from the site module'
    assert {k: net._PERLMUTTER_DEFAULTS.get(k) for k in network} == network
