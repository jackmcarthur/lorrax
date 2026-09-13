#!/usr/bin/env python3
"""Apply the version-checked optional experimental import fix to jax-xc 0.0.11.

Pass an explicit installation target; never discovers or changes the runtime.
Numerical functional code and global JAX APIs are untouched.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('site', type=Path)
a = p.parse_args()
init = a.site / 'jax_xc/__init__.py'
expected = 'fb636b487e69473bed92291819ce67395d106069f57d6a06e4a946122d5deeb0'
if hashlib.sha256(init.read_bytes()).hexdigest() != expected:
    raise SystemExit('Refusing: expected original jax-xc 0.0.11 package initializer')
patch = Path(__file__).with_name('jax_xc_lazy_experimental.patch').resolve()
subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(patch)],
               cwd=a.site, check=True)
print('Applied jax-xc 0.0.11 lazy-experimental patch; generated functionals unchanged')
