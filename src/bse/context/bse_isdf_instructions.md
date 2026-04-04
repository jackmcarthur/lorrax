# ISDF‑BSE Matrix‑Vector Product: Debugging Instructions for LLM Agents

This document guides an LLM agent through the **ISDF‑BSE** matrix‑vector application, assuming an iterative diagonalization solver is in place. It details the intended computation steps, pseudocode for all key multiplications, JAX sharding strategies, and explicit memory/compute bottlenecks.

---

## 1. Overview

We apply the BSE Hamiltonian $H_{
m BSE}=D + V - W$ to a vector $X_{i_v i_c k}$ in the following stages:

1. **Diagonal term** $D X$
2. **Separable Coulomb term** $V X$ using ISDF interpolants $\{\mu,
   u\}$
3. **Screened exchange term** $W X$ via convolution in $k$-space

Each stage must run at scale ($N_{v,c}\sim10^3$, $N_\mu\sim5\!	imes\!10^4$, $	ext{nk}\sim200^3$) on tens of GPUs with **minimal memory footprint** and **maximal locality**.

---

## 2. Device Mesh & Axis Names

Define a mesh over your GPU pool:

```python
import numpy as np
import jax
from jax.sharding import Mesh

# Example: 32 GPUs → choose sharding factors
P_i, P_mu, P_nu, P_kx, P_ky, P_kz = 2, 4, 4, 2, 2, 1
devices = np.array(jax.devices()[:32]).reshape((P_i, P_mu, P_nu, P_kx, P_ky, P_kz))
mesh = Mesh(devices, ('i','μ','ν','kx','ky','kz'))
```

* \`\`: excitation index $i_v i_c$
* \`\`: ISDF interpolation points
* \`\`: 3D $k$-grid

Other axes (spin, band sub-index, time) are treated as either small replicated dims (via `None`) or folded into `i`.

---

## 3. Pseudocode with JAX Shardings

### 3.1. Diagonal Term

```python
from jax import pjit
from jax.sharding import PartitionSpec as P

@pjit(
  in_shardings=P('i','kx','ky','kz'),
  out_shardings=P('i','kx','ky','kz')
)
def apply_D(X, eps_v, eps_c):
    # X[i,kx,ky,kz] holds X_{i_v i_c}(k)
    # eps_v,eps_c are replicated or small
    return (eps_v - eps_c)[...,None,None,None] * X
```

```python
@pjit(
  in_shardings=(P('μ','ν'),                  # V[μ,ν]
               P('i','kx','ky','kz','μ')),   # zeta_mu = u*(k,μ) * X-projection
  out_shardings=P('i','kx','ky','kz','ν')
)
def apply_V(V, zeta_mu):
    # mat-vec over μ→ν
    return jnp.einsum('μν,...μ->...ν', V, zeta_mu)
```

**Step A: build intermediate** $A_{i
u}(k)=\sum_\mu \overline u_i(k,μ)\,\zeta(μ,ν)$

```python
@pjit(
  in_shardings=(P('i','kx','ky','kz','μ'),  # ψ*(k,μ)
               P('μ','ν'),                  # ζ[μ,ν]
               P('i','kx','ky','kz','μ')), # ψ(k,μ)
  out_shardings=P('i','kx','ky','kz','ν')   # A[i,k,ν]
)
def build_A(psi_star, zeta, psi):
    tmp = psi_star * psi                     # local µ-block
    return jnp.einsum('...μ,μν->...ν', tmp, zeta)
```

**Step B: 3D FFT‐convolution** (local in k-space)

```python
from jax.numpy.fft import fftn, ifftn

@pjit(
  in_shardings=P('i','kx','ky','kz','ν'),  # A(k,ν)
  out_shardings=P('i','kx','ky','kz','ν')   # W X
)
def apply_W(A):
    A_k = fftn(A, axes=(1,2,3))
    B_k = A_k * Wq_k  # pre-sharded same
    return ifftn(B_k, axes=(1,2,3))
```

---

## 4. Memory & Compute Bottlenecks

* **Worst‐case memory** is storing `ψ(k,μ)` and `zeta(μ,ν)` shards when $N_\mu$ is largest → careful to shard heavily in `μ`.
* **Compute hotspot** is the local QR/solve for ISDF ($O(N_\mu^3/P_\mu)$) and the 3D FFT ($O(N_k \log N_k)$).
* **Minimal communication**:

  * One `all_gather` or `psum` per small contraction ($\mu	o
    u$).
  * Zero cross‑GPU traffic in the large $k$‑space FFT.

---

**Use this as a checklist** for your LLM agent when generating or debugging new kernels. Ensure all PartitionSpecs match these axes and that no full‑matrix gathers ever occur on the giant $(i,k)$ dims.
