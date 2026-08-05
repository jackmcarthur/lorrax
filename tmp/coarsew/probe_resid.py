import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from bse.bse_io import pad_W_R_to_grid, decimate_W_q_to_subgrid
def ifftn(x): return jnp.fft.ifftn(x, axes=(-3,-2,-1), norm="ortho")
def fftn(x):  return jnp.fft.fftn(x, axes=(-3,-2,-1), norm="ortho")
rng=np.random.default_rng(0)
for cg,fg in [((6,6,1),(12,12,1)), ((3,3,1),(6,6,1)), ((4,4,2),(8,8,4))]:
    Wq=jnp.asarray(rng.standard_normal((8,8,*cg))+1j*rng.standard_normal((8,8,*cg)))
    WRf=pad_W_R_to_grid(ifftn(Wq),fg)
    Wqf=fftn(WRf)
    on=decimate_W_q_to_subgrid(Wqf,cg)
    err=float(jnp.max(jnp.abs(on-Wq)))
    rel=err/float(jnp.max(jnp.abs(Wq)))
    print(f"{cg}->{fg}: abs_resid={err:.3e}  rel={rel:.3e}  scale=sqrt({int(np.prod(fg))}/{int(np.prod(cg))})={np.sqrt(np.prod(fg)/np.prod(cg)):.4f}")
WR=jnp.asarray(rng.standard_normal((3,3,12,12,1)))
print("noop byte-identical:", np.array_equal(np.asarray(pad_W_R_to_grid(WR,(12,12,1))), np.asarray(WR)))
