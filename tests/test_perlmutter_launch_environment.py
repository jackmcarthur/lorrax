"""Perlmutter setup selects future-step defaults without rewriting a live step.

These are plain host tests. They source the real Bash setup and evaluate the
real Lua descriptor; no JAX, MPI, GPU, scheduler, or allocation is required.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ('login', {}, False),
    ('allocation', {'SLURM_JOB_ID': '123', 'SLURM_JOB_NUM_NODES': '4'}, False),
    ('batch', {'SLURM_STEP_ID': 'batch'}, False),
    ('extern', {'SLURM_STEP_ID': 'extern'}, False),
    ('interactive', {'SLURM_STEP_ID': 'interactive'}, False),
    ('first-step', {'SLURM_STEP_ID': '0'}, True),
    ('later-step', {'SLURM_STEP_ID': '42'}, True),
    ('legacy-first', {'SLURM_STEPID': '0'}, True),
    ('legacy-later', {'SLURM_STEPID': '42'}, True),
    ('mixed-alias', {'SLURM_STEP_ID': 'batch', 'SLURM_STEPID': '3'}, True),
    ('mixed-modern', {'SLURM_STEP_ID': '3', 'SLURM_STEPID': 'batch'}, True),
]


def launch_env(step_env, network):
    """Isolate the subprocess from the parent test runner's launch context."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('SLURM_'):
            env.pop(key)
    env.update(step_env)
    if network is not None:
        env['SLURM_NETWORK'] = network
    return env


def assert_network(stdout, expected):
    assert stdout == ('\n\n' if expected is None else 'x\n'+expected+'\n')


@pytest.mark.parametrize('label,step_env,active', STEPS, ids=[x[0] for x in STEPS])
@pytest.mark.parametrize('network', [None, '', 'job_vni', 'no_vni,tcs=LOW_LATENCY'])
def test_gpu_setup_before_srun(tmp_path, label, step_env, active, network):
    env = launch_env(step_env, network)
    # This setup test does not probe the CUDA wheel or create a cudart shim.
    python = tmp_path/'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    env.update(VIRTUAL_ENV=str(python.parent.parent), SCRATCH=str(tmp_path))
    result = subprocess.run([
        'bash', '--noprofile', '--norc', '-c',
        'source "$1"; source "$1"; printf "%s\\n" "${SLURM_NETWORK+x}" "${SLURM_NETWORK-}"',
        'bash', str(ROOT/'config/perlmutter/gpu_env.sh')],
        env=env, text=True, capture_output=True, check=True)
    expected = network if network is not None or active else 'no_vni'
    assert_network(result.stdout, expected)


@pytest.mark.parametrize('label,step_env,active', STEPS, ids=[x[0] for x in STEPS])
@pytest.mark.parametrize('network', [None, '', 'job_vni', 'no_vni,tcs=LOW_LATENCY'])
@pytest.mark.parametrize('module_path', [
    'config/modulefiles/lorrax/0.1.0.lua',
    'config/perlmutter/module/module.lua.in'])
def test_module_setup_before_srun(label, step_env, active, network, module_path):
    lua = shutil.which('lua')
    if not lua:
        pytest.skip('Lua interpreter is unavailable')
    # Lmod capability declarations are irrelevant here. Execute the actual
    # descriptor's Lua control flow and observe only its environment effects.
    program = r'''
local assigned = {}
help = function() end
whatis = function() end
family = function() end
depends_on = function() end
prepend_path = function() end
pathJoin = function(...) return table.concat({...}, "/") end
myFileName = function() return arg[1] end
setenv = function(name, value) assigned[name] = value end
pushenv = setenv
dofile(arg[1])
local value = assigned["SLURM_NETWORK"]
if value == nil then value = os.getenv("SLURM_NETWORK") end
io.write(value == nil and "" or "x", "\n", value or "", "\n")
'''
    result = subprocess.run([lua, '-', str(ROOT/module_path)],
        input=program, env=launch_env(step_env, network),
        text=True, capture_output=True, check=True)
    expected = network if network is not None or active else 'no_vni'
    assert_network(result.stdout, expected)


@pytest.mark.parametrize('step', [None, 'batch', '0'])
@pytest.mark.parametrize('network', [None, '', 'job_vni'])
def test_lmod_unload_restores_network_choice(step, network):
    lmod = os.environ.get('LMOD_CMD')
    if not lmod or not os.access(lmod, os.X_OK):
        pytest.skip('Lmod is unavailable')
    env = launch_env({} if step is None else {'SLURM_STEP_ID': step}, network)
    # A fresh module state in the child avoids inheriting a loaded installation.
    for name in list(env):
        if name.startswith('_ModuleTable') or name in (
                'LOADEDMODULES', '_LMFILES_', 'LMOD_FAMILY_LORRAX'):
            env.pop(name)
    program = r'''
eval "$("$2" bash use "$1/config/modulefiles")"
eval "$("$2" bash load lorrax/0.1.0)"
printf '%s\n' "${SLURM_NETWORK+x}" "${SLURM_NETWORK-}"
eval "$("$2" bash unload lorrax/0.1.0)"
printf '%s\n' "${SLURM_NETWORK+x}" "${SLURM_NETWORK-}"
'''
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', program,
        'bash', str(ROOT), lmod], env=env, text=True, capture_output=True, check=True)
    loaded = network if network is not None or step == '0' else 'no_vni'
    def record(value):
        return '\n\n' if value is None else 'x\n'+value+'\n'
    assert result.stdout == record(loaded)+record(network)
