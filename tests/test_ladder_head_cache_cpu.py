"""A head matvec cannot cross meshes sharing only their shape."""
import os
from pathlib import Path
import subprocess
import sys


def _controls():
    from unittest.mock import patch
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from distrib_la import mesh_key
    from bse import head_resolvent as owner

    devices = np.asarray(jax.devices('cpu'))
    first = Mesh(devices.reshape(2,2),('x','y'))
    same = Mesh(devices.copy().reshape(2,2),('x','y'))
    other = Mesh(devices[[0,2,1,3]].reshape(2,2),('x','y'))
    data = dict(nkx=1,nky=1,nkz=1,n_cond_pad=2,n_val_pad=2)

    def build(mesh, _data, **_kwargs):
        # The production builder closes over its mesh. No solve or large
        # payload is needed to test which closure this cache returns.
        return (lambda _: mesh_key(mesh)), None, None, None, None

    with patch.object(owner,'build_ladder_resolvent',build), \
         patch.object(owner,'_HEAD_MATVEC_CACHE',{}), \
         patch.object(owner,'_HEAD_OPERATOR_IDS',{}):
        a = owner.build_head_operator(first,data)
        repeated = owner.build_head_operator(same,data)
        b = owner.build_head_operator(other,data)
        assert a.key == repeated.key and a.matvec is repeated.matvec
        assert a.key[-1] == mesh_key(first)
        assert a.key != b.key and a.matvec is not b.matvec
        assert b.matvec(None) == mesh_key(other)


def test_head_matvec_cache_binds_complete_mesh_identity():
    env = dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1')
    flags = [flag for flag in env.get('XLA_FLAGS','').split()
             if not flag.startswith('--xla_force_host_platform_device_count')]
    env['XLA_FLAGS'] = ' '.join(flags+['--xla_force_host_platform_device_count=4'])
    result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--controls'],
                            env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+'\n'+result.stderr


if __name__ == '__main__':
    _controls()
