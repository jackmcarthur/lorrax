import os, sys
sys.path.insert(0, 'src')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from solvers.sternheimer_solve import sternheimer_solve, SternheimerOp

nG = 6
nv = 2
nspinor = 1
nx=ny=nz=4

key = jax.random.PRNGKey(0)
T_diag = jnp.abs(jax.random.normal(key, (nG,))) + 2.0  # positive definite kinetic
V_scf = jnp.zeros((nx,ny,nz))
Gx = jnp.zeros(nG, dtype=jnp.int32)
Gy = jnp.zeros(nG, dtype=jnp.int32)
Gz = jnp.zeros(nG, dtype=jnp.int32)
vnl_Z = jnp.zeros((0, nG), dtype=jnp.complex128)
vnl_E = jnp.zeros((nspinor, nspinor, 0, 0), dtype=jnp.complex128)
mask = jnp.ones(nG)
U_val = jnp.zeros((0, nspinor, nG), dtype=jnp.complex128)  # no P_val projector -> alpha_pv term vanishes regardless
eps_v = jnp.array([0.1, 0.2])
alpha_pv = jnp.asarray(0.0)
precond_diag = jnp.ones((nv,1,nG))
fft_grid = (nx,ny,nz)

op = SternheimerOp(T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask,
                    U_val, eps_v, alpha_pv, precond_diag, fft_grid)

key, sub = jax.random.split(key)
b = (jax.random.normal(sub, (nv, nspinor, nG)) + 1j*jax.random.normal(sub, (nv, nspinor, nG))).astype(jnp.complex128)

# primal check: A x = -b  =>  x = -A^{-1} b ; A = diag(T_diag - eps_v)
A_diag = T_diag[None,None,:] - eps_v[:,None,None]  # (nv,1,nG)
x_expected = -b / A_diag
x = sternheimer_solve(op, b, tol=1e-12, max_iter=500)
print('primal max err:', jnp.max(jnp.abs(x - x_expected)))

# JVP wrt b only, op fixed (op_dot = zero pytree)
def f_b(bb):
    return sternheimer_solve(op, bb, tol=1e-12, max_iter=500)

key, sub2 = jax.random.split(key)
db = (jax.random.normal(sub2, (nv, nspinor, nG)) + 1j*jax.random.normal(sub2, (nv, nspinor, nG))).astype(jnp.complex128)

x0, xdot = jax.jvp(f_b, (b,), (db,))
xdot_expected = -db / A_diag   # since x = -A^{-1} b linear in b => xdot = -A^{-1} db
print('jvp max err vs expected (-A^-1 db):', jnp.max(jnp.abs(xdot - xdot_expected)))
print('jvp max err vs flipped (+A^-1 db):', jnp.max(jnp.abs(xdot - (+db/A_diag))))

# finite difference check
eps = 1e-6
xp = f_b(b + eps*db)
xm = f_b(b - eps*db)
xdot_fd = (xp - xm) / (2*eps)
print('fd vs expected (-A^-1 db):', jnp.max(jnp.abs(xdot_fd - xdot_expected)))
print('fd vs autodiff jvp:', jnp.max(jnp.abs(xdot_fd - xdot)))
