# Parallel Algorithms for ISDF-BSE Matrix-Vector Products (v2)

## 1. Problem Setup and Notation

We work in the Tamm-Dancoff approximation where $H_\text{BSE}(\mathbf{Q}) = D(\mathbf{Q}) + 2V_A^\mathbf{Q} - W_A^\mathbf{Q}$, generalized to finite exciton momentum $\mathbf{Q}$.

### 1.1 Finite-Q BSE kernel structure

At exciton momentum $\mathbf{Q}$, the eigenvector $X^\mathbf{Q}(v,c,\mathbf{k})$ describes the electron at $(c,\mathbf{k}+\mathbf{Q})$ and hole at $(v,\mathbf{k})$.

**Diagonal:**
$$[D(\mathbf{Q})\,X](i_v,i_c,\mathbf{k}) = (\epsilon_{i_c,\mathbf{k}+\mathbf{Q}} - \epsilon_{i_v,\mathbf{k}})\,X(i_v,i_c,\mathbf{k})$$

**Direct (W) in ISDF** — the $\mathbf{Q}$'s cancel in the phase:
$$W_A^\mathbf{Q}(i_v i_c \mathbf{k},\; j_v j_c \mathbf{k}') = \frac{1}{N_k}\sum_{\mu,\nu} \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu)\, u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_\mu)\;\tilde{W}_{\mathbf{k}-\mathbf{k}',\mu\nu}\; \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_\nu)\, u_{i_v,\mathbf{k}}(\hat{r}_\nu)$$

The convolution kernel $\tilde{W}_{\mathbf{q},\mu\nu}$ is **independent of Q**. Conduction wavefunctions enter Q-shifted; valence wavefunctions are unshifted.

**Exchange (V) in ISDF** — acquires a Q-dependent kernel:
$$V_A^\mathbf{Q}(i_v i_c \mathbf{k},\; j_v j_c \mathbf{k}') = \frac{1}{N_k}\sum_{\mu,\nu} \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu)\, u_{i_v,\mathbf{k}}(\hat{r}_\mu)\;\tilde{V}^\mathbf{Q}_{\mu\nu}\; \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_\nu)\, u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_\nu)$$

where $\tilde{V}^\mathbf{Q}_{\mu\nu} = |\Omega|\sum_\mathbf{G} \frac{4\pi}{|\mathbf{Q}+\mathbf{G}|^2}\,\overline{\hat{\zeta}^V_\mu}(\mathbf{G})\,\hat{\zeta}^V_\nu(\mathbf{G})$.

At $\mathbf{Q}=0$: $V^\mathbf{Q}$ reduces to the standard $\tilde{V}_{\mu\nu}$ with the $\mathbf{G}=0$ term excluded (per Eq. 2-32 of Henneke), and all conduction wavefunctions revert to unshifted momenta.

### 1.2 The ISDF Matvec

**W piece** (Henneke Eq. 4-6, generalized to finite Q):
$$[W_A^\mathbf{Q} X](i_v,i_c,\mathbf{k}) = \frac{1}{N_k}\sum_\nu u_{i_v,\mathbf{k}}(\hat{r}_\nu) \sum_\mu \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu) \sum_{\mathbf{k}'} \tilde{W}_{\mathbf{k}-\mathbf{k}',\mu\nu} \sum_{j_c} u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_\mu) \sum_{j_v} \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_\nu)\, X(j_v,j_c,\mathbf{k}')$$

**V piece** (Henneke Eq. 4-5, generalized):
$$[V_A^\mathbf{Q} X](i_v,i_c,\mathbf{k}) = \frac{1}{N_k}\sum_\mu \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu)\, u_{i_v,\mathbf{k}}(\hat{r}_\mu) \sum_\nu \tilde{V}^\mathbf{Q}_{\mu\nu}\,\underbrace{\sum_{\mathbf{k}'}\sum_{j_c} u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_\nu)\sum_{j_v} \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_\nu)\, X(j_v,j_c,\mathbf{k}')}_{a^\mathbf{Q}(\nu)}$$

### 1.3 Wavefunction convention

Throughout, conduction wavefunctions always appear Q-shifted:

| Step | Cond. wfn evaluated at | Val. wfn evaluated at |
|------|------------------------|-----------------------|
| W forward (jv,jc contract) | $u_{j_c,\mathbf{k}'+\mathbf{Q}}(\mu)$ | $u_{j_v,\mathbf{k}'}(\nu)$ |
| W reverse (iv,ic expand) | $u_{i_c,\mathbf{k}+\mathbf{Q}}(\mu)$ | $u_{i_v,\mathbf{k}}(\nu)$ |
| V forward (jv,jc contract) | $u_{j_c,\mathbf{k}'+\mathbf{Q}}(\nu)$ | $u_{j_v,\mathbf{k}'}(\nu)$ |
| V reverse (iv,ic expand) | $u_{i_c,\mathbf{k}+\mathbf{Q}}(\mu)$ | $u_{i_v,\mathbf{k}}(\mu)$ |

At $\mathbf{Q}=0$ all entries collapse to unshifted momenta. Since all bands are stored at every k-point and at local ISDF points, the Q-shift is simply a different k-index lookup with no structural change to the parallelization.

### 1.4 Sizes (many-band regime)

The target regime has $N_v, N_c \sim 400$, $N_\mu \sim 200$, $N_k \sim 16\text{--}2000$.

| Quantity | Shape | Scaling note |
|----------|-------|-------------|
| $X(v,c,\mathbf{k})$ | $N_v \times N_c \times N_k$ | **Dominant** when $N_v N_c \gg N_\mu^2$ |
| $\tilde{W}_\mathbf{R}(\mu,\nu)$ | $N_k \times N_\mu \times N_\mu$ | Cannot be replicated |
| $u_{n\mathbf{k}}(\hat{r}_\mu)$ | $(N_v{+}N_c) \times N_k \times N_\mu$ | Moderate |
| $\tilde{V}^\mathbf{Q}(\mu,\nu)$ | $N_\mu^V \times N_\mu^V$ | Tiny, replicable |

With $N_v N_c = 160\text{K} \gg N_\mu^2 = 40\text{K}$, the Krylov vectors dominate memory and **must** be distributed.

---

## 2. Scheme: Distributed $X(v_Y, c_X, \mathbf{k})$ on the 2D Grid

### 2.1 Core idea

We place both the ISDF index AND the band index for each kernel leg on the **same** processor axis:

- **X-axis** owns μ (ISDF) and c (conduction bands)
- **Y-axis** owns ν (ISDF) and v (valence bands)

W is distributed as before over (μ_X, ν_Y). The Krylov vector $X(v_Y, c_X, \mathbf{k})$ is distributed over BOTH axes, reducing per-proc storage by the full factor $P$.

The forward band contractions (building $B$ from $X$) require **ring communication** to expose all band chunks to each ISDF point, since a proc at position $(x,y)$ has wavefunctions for all bands at its local ISDF points, but only its local slice of $X$. The reverse contractions (expanding $C$ back to band space) are standard **reduce-scatters**.

### 2.2 Data layout

Processor grid: $P = P_X \times P_Y$.

| Data | Sharding | Size/proc |
|------|----------|-----------|
| $\psi_{n,\mathbf{k}}(\hat{r}_{\mu_X})$ | **All** $n$ (all bands), μ sharded over X; repl. over Y | $(N_v{+}N_c)\cdot N_k \cdot N_\mu/P_X$ |
| $\psi_{n,\mathbf{k}}(\hat{r}_{\nu_Y})$ | **All** $n$ (all bands), ν sharded over Y; repl. over X | $(N_v{+}N_c)\cdot N_k \cdot N_\mu/P_Y$ |
| $W_\mathbf{R}(\mu_X, \nu_Y)$ | 2D block; pre-FFT'd from $W_\mathbf{q}$ | $N_k \cdot N_\mu^2 / P$ |
| $\tilde{V}^\mathbf{Q}(\mu,\nu)$ | Replicated (tiny) | $(N_\mu^V)^2$ |
| $X(v_Y, c_X, \mathbf{k})$ | v over Y, c over X | $N_v N_c N_k / P$ |

Each proc stores wavefunctions for **all bands** at its local ISDF points, in two copies (one μ-sharded, one ν-sharded). This is the key enabler: any band contraction can be computed locally at local ISDF points, given the appropriate slice of $X$ or intermediate.

### 2.3 Why forward contractions are rings, not allreduces

Consider building $A(\nu_Y, c_X, \mathbf{k}') = \sum_v \bar{u}_{v,\mathbf{k}'}(\nu_Y)\, X(v, c_X, \mathbf{k}')$. Each proc at position $(x,y)$ has:
- $\bar{u}_v(\nu_Y)$ for ALL $v$ at its local $\nu_Y$ points
- $X(v_Y, c_X, \mathbf{k}')$ for only its local $v_Y$ chunk

The partial-$v_Y$ sum at local $\nu_Y$ gives a contribution to $A$, but to complete the sum over ALL $v$, the proc needs $X$ at other $v$ chunks. Each proc in the Y-group has $\psi$ at **different** $\nu_Y$ points, so an allreduce (which sums same-indexed elements across procs) would incorrectly mix contributions at different $\nu$ points.

The correct operation is to **circulate** $X$ chunks around the Y-ring so each proc can evaluate the full $v$-sum at its own $\nu_Y$ using its own wavefunctions. Identically, the $c$-contraction requires circulating intermediates around the X-ring.

---

## 3. W Matvec: $[W_A^\mathbf{Q} X^\mathbf{Q}](i_v, i_c, \mathbf{k})$

### Step 1 — Valence contraction (ring over Y)

Target: $A_{\mathbf{k}'}(\nu_Y, c_X) = \sum_{j_v} \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_{\nu_Y})\; X^\mathbf{Q}(j_v, c_X, \mathbf{k}')$

Each proc needs the full $v$-sum but only has $X$ for its local $v_Y$. Ring over Y, $P_Y$ steps:

```
A(ν_Y, c_X, k') = 0
buf = X(v_Y, c_X, k')                                           # local slice
for t = 0, ..., P_Y - 1:
    v_chunk = indices owned by (my_y + t) mod P_Y
    A(ν_Y, c_X, k') += ψ*_{v_chunk, k'}(ν_Y) · buf(v_chunk, c_X, k')    # GEMM
    send buf → Y_right;  recv buf ← Y_left
```

- Ring buffer: $(N_v/P_Y) \times (N_c/P_X) \times N_k = N_v N_c N_k / P$ — **one Krylov vector slice**
- Each GEMM: $(N_\mu/P_Y) \times (N_v/P_Y) \times (N_c/P_X \cdot N_k)$
- FLOPs/proc: $N_\mu N_v N_c N_k / P$ (same total as non-distributed)
- Comm: $(P_Y{-}1) \times N_v N_c N_k / P$ (ring bandwidth)
- Result: $A(\nu_Y, c_X, \mathbf{k}')$ of shape $(N_\mu/P_Y, N_c/P_X, N_k)$ — fully assembled

### Step 2 — Conduction contraction (ring over X)

Target: $B_{\mathbf{k}'}(\mu_X, \nu_Y) = \sum_{j_c} u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_{\mu_X})\; A_{\mathbf{k}'}(\nu_Y, j_c)$

$A$ from step 1 has shape $(\nu_Y, c_X, \mathbf{k}')$ — local in both $\nu_Y$ and $c_X$. But the sum over $j_c$ requires $A$ at ALL $c$ values. Ring over X, circulating $A$:

```
B(μ_X, ν_Y, k') = 0
buf = A(ν_Y, c_X, k')                                           # from step 1
for t = 0, ..., P_X - 1:
    c_chunk = indices owned by (my_x + t) mod P_X
    B(μ_X, ν_Y, k') += ψ_{c_chunk, k'+Q}(μ_X) · buf(ν_Y, c_chunk, k')  # GEMM
    send buf → X_right;  recv buf ← X_left
```

- Ring buffer: $(N_\mu/P_Y) \times (N_c/P_X) \times N_k = N_\mu N_c N_k / P$
- FLOPs/proc: $N_\mu^2 N_c N_k / P$ (same total)
- Comm: $(P_X{-}1) \times N_\mu N_c N_k / P$
- Result: $B(\mu_X, \nu_Y, \mathbf{k}')$ of shape $(N_\mu/P_X, N_\mu/P_Y, N_k)$ — fully assembled

### Step 3 — FFT convolution (fully local)

$$B_\mathbf{R}(\mu_X, \nu_Y) = \text{FFT}_{\mathbf{k}'\to\mathbf{R}}[B_{\mathbf{k}'}(\mu_X, \nu_Y)]$$
$$C_\mathbf{R}(\mu_X, \nu_Y) = W_\mathbf{R}(\mu_X, \nu_Y) \odot B_\mathbf{R}(\mu_X, \nu_Y) \qquad\text{(Hadamard)}$$
$$C_\mathbf{k}(\mu_X, \nu_Y) = \text{IFFT}_{\mathbf{R}\to\mathbf{k}}[C_\mathbf{R}(\mu_X, \nu_Y)]$$

- **Zero communication.** $W_\mathbf{R}$ and $B_\mathbf{R}$ have matching local shapes.
- FLOPs/proc: $O(N_\mu^2 N_k \log N_k / P)$
- Pre-store $W_\mathbf{R}$ (one-time FFT of $W_\mathbf{q}$, amortized over all iterations).

### Step 4 — Conduction expansion (reduce-scatter over X)

Target: $D_\mathbf{k}(c_X, \nu_Y) = \sum_\mu \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu)\; C_\mathbf{k}(\mu, \nu_Y)$

Each proc computes the partial-$\mu_X$ contribution for **all** $c$ (it has all bands at local $\mu_X$):

$$D_\mathbf{k}(c, \nu_Y)\{U_X\} = \sum_{\mu_X} \bar{u}_{c,\mathbf{k}+\mathbf{Q}}(\hat{r}_{\mu_X})\; C_\mathbf{k}(\mu_X, \nu_Y)$$

Temporary shape: $(N_c, N_\mu/P_Y, N_k)$. This is a partial-$\mu$ sum that must be completed by summing across the X-group. Each proc needs only its own $c_X$ slice of the final result.

**Reduce-scatter over X:** each proc contributes its partial sum; the reduce-scatter sums over all $\mu$ slices and delivers each proc its $c_X$ slice.

- Temp: $N_c \times (N_\mu/P_Y) \times N_k$ per proc (before reduce-scatter)
- Comm: $\frac{P_X - 1}{P_X} \times N_c N_\mu N_k / P_Y$ per proc
- Result: $D(c_X, \nu_Y, \mathbf{k})$ of shape $(N_c/P_X, N_\mu/P_Y, N_k)$

If the temporary is too large, chunk over $c$ in blocks of size $N_c/P_X$ and do $P_X$ reduce-scatters sequentially (same total comm, bounded temp).

### Step 5 — Valence expansion (reduce-scatter over Y)

Target: $[W_A^\mathbf{Q} X](v_Y, c_X, \mathbf{k}) = \sum_\nu u_{i_v,\mathbf{k}}(\hat{r}_\nu)\; D_\mathbf{k}(c_X, \nu)$

Each proc computes partial-$\nu_Y$ contribution for **all** $v$:

$$[W_A^\mathbf{Q} X](v, c_X, \mathbf{k})\{U_Y\} = \sum_{\nu_Y} u_{v,\mathbf{k}}(\hat{r}_{\nu_Y})\; D_\mathbf{k}(c_X, \nu_Y)$$

**Reduce-scatter over Y:** sums over $\nu$, delivers each proc its $v_Y$ slice.

- Temp: $N_v \times (N_c/P_X) \times N_k$ per proc
- Comm: $\frac{P_Y-1}{P_Y} \times N_v N_c N_k / P_X$ per proc
- Result: $[W_A^\mathbf{Q} X](v_Y, c_X, \mathbf{k})$ — matches Krylov vector layout ✓

---

## 4. V Matvec: $[V_A^\mathbf{Q} X^\mathbf{Q}](i_v, i_c, \mathbf{k})$

### Step 1v — Build $a^\mathbf{Q}(\nu_Y)$ (fused with W step 1 ring)

Target: $a^\mathbf{Q}(\nu_Y) = \sum_{\mathbf{k}', j_c, j_v} u_{j_c,\mathbf{k}'+\mathbf{Q}}(\hat{r}_{\nu_Y})\; \bar{u}_{j_v,\mathbf{k}'}(\hat{r}_{\nu_Y})\; X(j_v, j_c, \mathbf{k}')$

This requires the full sum over both $v$ and $c$, but X has $v_Y$ and $c_X$.

**Fuse with the W step 1 ring over Y**: during the same ring circulation of X chunks, accumulate $a^\mathbf{Q}$ alongside $A$. At each ring step, for each arriving $v$-chunk:

$$a^\mathbf{Q}(\nu_Y)\{U_X\} \mathrel{+}= \sum_{\mathbf{k}'}\sum_{j_c \in c_X} u_{j_c,\mathbf{k}'+\mathbf{Q}}(\nu_Y) \sum_{j_v \in v_\text{chunk}} \bar{u}_{j_v,\mathbf{k}'}(\nu_Y)\; X(v_\text{chunk}, c_X, \mathbf{k}')$$

After the ring completes: $a$ has full $v$-sum but partial $c$-sum ($c_X$ only).

**Allreduce $a$ over X** to complete the $c$-sum. Volume: $N_\mu^V / P_Y$. **Negligible.**

(This allreduce IS correct: all procs in the X-group share the same $\nu_Y$ indices, so the element-wise sum correctly accumulates different $c_X$ contributions to the same $a(\nu_Y)$ values.)

### Step 2v — Apply $\tilde{V}^\mathbf{Q}$

$$t^\mathbf{Q}(\mu) = \sum_\nu \tilde{V}^\mathbf{Q}_{\mu\nu}\; a^\mathbf{Q}(\nu)$$

$a$ has shape $N_\mu^V / P_Y$ per proc (local $\nu$). Partial-$\nu$ sum:

$$t^\mathbf{Q}(\mu)\{U_Y\} = \sum_{\nu_Y} \tilde{V}^\mathbf{Q}(\mu, \nu_Y)\; a(\nu_Y)$$

**Allreduce $t$ over Y.** Volume: $N_\mu^V$. **Negligible.**

After allreduce, every proc has the full $t^\mathbf{Q}(\mu)$.

### Step 3v — Expand to $(v_Y, c_X)$ (reduce-scatter over X)

Each proc computes the partial-$\mu_X$ contribution for all $c$ at local $v_Y$:

$$[V_A^\mathbf{Q} X](v_Y, c, \mathbf{k})\{U_X\} = \sum_{\mu_X} \bar{u}_{c,\mathbf{k}+\mathbf{Q}}(\mu_X)\; u_{v_Y,\mathbf{k}}(\mu_X)\; t^\mathbf{Q}(\mu_X)$$

**Reduce-scatter over X:** delivers each proc its $c_X$ slice.

- Temp: $(N_v/P_Y) \times N_c \times N_k$
- Comm: $\frac{P_X-1}{P_X} \times N_v N_c N_k / P_Y$
- Result: $[V_A^\mathbf{Q} X](v_Y, c_X, \mathbf{k})$ — matches Krylov vector layout ✓

---

## 5. Combined Algorithm Summary

### Pseudocode

```
# Data on mesh (X, Y):
ψ_X[n, k, μ_X]       — all bands, μ sharded over X, replicated over Y
ψ_Y[n, k, ν_Y]       — all bands, ν sharded over Y, replicated over X
W_R[R, μ_X, ν_Y]     — 2D block, pre-FFT'd
V_Q[μ, ν]             — replicated (tiny)
X[v_Y, c_X, k]        — v over Y, c over X

# ===== W PIECE =====

# Step 1: v-contraction — RING over Y
#   Circulate X chunks; each step, GEMM with local ψ_Y
#   Simultaneously accumulate V's a(ν) vector (fused V step 1v)
A[ν_Y, c_X, k'], a_partial[ν_Y] = ring_accumulate_Y(
    buf_init = X[v_Y, c_X, k],
    body_W = λ(v_chunk, buf): ψ_Y*[v_chunk, k'] @ buf[v_chunk, c_X, k']
    body_V = λ(v_chunk, buf): Σ_k',c_X ψ_Y[c_X,k'+Q] · ψ_Y*[v_chunk,k'] · buf
)

# V step 1v completion: allreduce a over X (tiny)
a[ν_Y] = allreduce_X(a_partial)

# Step 2: c-contraction — RING over X
#   Circulate A chunks; each step, GEMM with local ψ_X
B[μ_X, ν_Y, k'] = ring_accumulate_X(
    buf_init = A[ν_Y, c_X, k'],
    body = λ(c_chunk, buf): ψ_X[c_chunk, k'+Q, μ_X] @ buf[ν_Y, c_chunk, k']
)

# Step 3: convolution — LOCAL
B_R = FFT_k[B];  C_R = W_R ⊙ B_R;  C_k = IFFT[C_R]

# Step 4: c-expansion — REDUCE-SCATTER over X
D_partial[c, ν_Y, k] = ψ_X*[c, k+Q, μ_X] @ C[μ_X, ν_Y, k]    # all c, local μ
D[c_X, ν_Y, k] = reduce_scatter_X(D_partial)

# Step 5: v-expansion — REDUCE-SCATTER over Y
WX_partial[v, c_X, k] = ψ_Y[v, k, ν_Y] @ D[c_X, ν_Y, k]       # all v, local ν
WX[v_Y, c_X, k] = reduce_scatter_Y(WX_partial)

# ===== V PIECE (remaining steps) =====

# Step 2v: apply V^Q — tiny allreduce
t_partial[μ] = V_Q[μ, ν_Y] · a[ν_Y]
t[μ] = allreduce_Y(t_partial)

# Step 3v: expand — REDUCE-SCATTER over X
VX_partial[v_Y, c, k] = ψ_X*[c, k+Q, μ_X] · ψ_X[v_Y, k, μ_X] · t[μ_X]
VX[v_Y, c_X, k] = reduce_scatter_X(VX_partial)

# ===== D PIECE =====
DX[v_Y, c_X, k] = (ε_{c_X, k+Q} - ε_{v_Y, k}) · X[v_Y, c_X, k]   # local

# ===== COMBINE =====
result[v_Y, c_X, k] = DX + 2·VX - WX
```

### Operation taxonomy

| Step | Direction | Primitive | Axis | Buffer size/proc |
|------|-----------|-----------|------|------------------|
| W-1 (v contract) | forward | **ring** | Y | $N_v N_c N_k / P$ |
| W-2 (c contract) | forward | **ring** | X | $N_\mu N_c N_k / P$ |
| W-3 (convolution) | — | local | — | 0 |
| W-4 (c expand) | reverse | **reduce-scatter** | X | $N_c N_\mu N_k / (P_X P_Y)$ |
| W-5 (v expand) | reverse | **reduce-scatter** | Y | $N_v N_c N_k / (P_X P_Y)$ |
| V-1 (build a) | forward | fused with W-1 ring + tiny allreduce | X | negligible |
| V-2 (apply V) | — | tiny allreduce | Y | negligible |
| V-3 (expand) | reverse | **reduce-scatter** | X | $N_v N_c N_k / (P_X P_Y)$ |

### Why forward = ring, reverse = reduce-scatter

**Forward direction** (band space → ISDF space): to build $B(\mu_X, \nu_Y, \mathbf{k})$, every proc needs the **complete** result at its local $(\mu_X, \nu_Y)$ block — this feeds the FFT convolution, which acts on the full k-grid pointwise in $(\mu,\nu)$. Completing the band sums requires seeing ALL $v$ and $c$ chunks. Since $\psi$ at different ISDF points lives on different procs, each proc must receive the $X$ (or intermediate $A$) data and contract it locally against its own wavefunctions. This is a ring-accumulate (or equivalently, allgather + local compute).

**Reverse direction** (ISDF space → band space): $C(\mu_X, \nu_Y, \mathbf{k})$ is complete on each proc. Each proc can compute the partial-$\mu$ (or $\nu$) contribution for ALL bands locally, since it has all bands at its local ISDF points. The output $X(v_Y, c_X, \mathbf{k})$ is distributed — each proc only needs its own slice. This is a reduce-scatter: sum the partial contributions across the axis and scatter each slice to its owner.

---

## 6. Communication Analysis

### Bandwidth per proc per matvec

| Step | Type | Volume/proc (complex numbers) |
|------|------|------|
| W-1: v-ring | ring, $P_Y$ passes | $(P_Y{-}1) \cdot \frac{N_v N_c N_k}{P}$ |
| W-2: c-ring | ring, $P_X$ passes | $(P_X{-}1) \cdot \frac{N_\mu N_c N_k}{P}$ |
| W-4: c reduce-scatter | RS over X | $\frac{P_X{-}1}{P_X} \cdot \frac{N_c N_\mu N_k}{P_Y}$ |
| W-5: v reduce-scatter | RS over Y | $\frac{P_Y{-}1}{P_Y} \cdot \frac{N_v N_c N_k}{P_X}$ |
| V-3: reduce-scatter | RS over X | $\frac{P_X{-}1}{P_X} \cdot \frac{N_v N_c N_k}{P_Y}$ |
| V-1,2: allreduces | allreduce | $\ll$ others |

For $P_X, P_Y \gg 1$, asymptotic total bandwidth per proc:

$$\text{BW}_\text{total} \approx \frac{N_v N_c N_k}{P_X} + \frac{N_\mu N_c N_k}{P_Y} + \frac{N_c N_\mu N_k}{P_X P_Y} + \frac{N_v N_c N_k}{P_X P_Y} + \frac{N_v N_c N_k}{P_X P_Y}$$

The **dominant terms** are the two ring steps: $\frac{N_v N_c N_k}{P_X} + \frac{N_\mu N_c N_k}{P_Y}$.

### Comparison with old scheme (replicated X)

The old 2D scheme with replicated $X$ had dominant comm from allreduces of the full output vector in W step 5 and V step 3: each allreduced $N_v N_c N_k$ entries over $P_Y$ or $P_X$ processors, costing $\sim 2 N_v N_c N_k$ bandwidth per proc per allreduce.

| | Old scheme | New scheme |
|---|---|---|
| Dominant comm/proc | $\sim 4 N_v N_c N_k$ | $\sim N_v N_c N_k / P_X + N_\mu N_c N_k / P_Y$ |
| **Ratio** | 1 | $\sim 1/(2P_X)$ |

**Concrete example:** $N_v{=}N_c{=}400$, $N_\mu{=}200$, $N_k{=}1000$, $P_X{=}P_Y{=}8$ ($P{=}64$):

| | Old (repl. X) | New (dist. X) |
|---|---|---|
| Dominant | $4 \times 160\text{M} = 640\text{M}$ | $20\text{M} + 10\text{M} = 30\text{M}$ |
| **Speedup** | **1** | **~20×** |

### Latency

| | Messages/iter | Group size |
|---|---|---|
| Old (4 allreduces over $\sqrt{P}$) | $\sim 4 \times 2 \times 7 = 56$ | 8 |
| New (2 rings + 3 RS) | $2(P_X{+}P_Y{-}2) + 3 \times 2 \times 7 \approx 56$ | 8 |

Comparable latency, dramatically less bandwidth.

---

## 7. Computation Summary

FLOPs per proc per matvec (unchanged from non-distributed):

| Step | FLOPs/proc |
|------|-----------|
| W-1 + W-5 (v contractions) | $\frac{2 N_\mu N_v N_c N_k}{P}$ |
| W-2 + W-4 (c contractions) | $\frac{2 N_\mu^2 N_c N_k}{P}$ |
| W-3 (FFT + Hadamard) | $\frac{N_\mu^2 N_k \log N_k}{P}$ |
| V (all steps) | $\frac{2 N_\mu^V N_v N_c N_k + (N_\mu^V)^2}{P}$ |

**In the many-band regime** ($N_v N_c \gg N_\mu^2$), steps 1+5 dominate:

$$\text{FLOPs/proc} \approx \frac{2(N_\mu + N_\mu^V) N_v N_c N_k}{P}$$

The ring structure adds zero extra FLOPs — every GEMM is the same total size, just decomposed into $P_Y$ (or $P_X$) sub-GEMMs executed as each chunk arrives. With proper pipelining (overlap ring send/recv with GEMM on current chunk), the effective communication overhead is minimal.

---

## 8. Memory Budget

### Per-processor memory

| Quantity | Expression | Example ($P_X{=}P_Y{=}8$, $P{=}64$) |
|----------|-----------|---------------------------|
| Krylov vectors | $n_\text{vec} \cdot N_v N_c N_k / P$ | $200 \times 2.5\text{M} \times 16\text{B} = 8$ GB |
| $W_\mathbf{R}$ | $N_k \cdot N_\mu^2 / P$ | $1000 \times 625 \times 16\text{B} = 10$ MB |
| ψ (2 copies) | $2(N_v{+}N_c) N_k N_\mu / P_X$ | $2 \times 800 \times 1000 \times 25 \times 16\text{B} = 640$ MB |
| Ring buffer | $N_v N_c N_k / P$ | $2.5\text{M} \times 16\text{B} = 40$ MB |
| Work arrays ($B$, $C$) | $N_\mu^2 N_k / P$ | reuse $W$ memory = 10 MB |
| RS temporary | $\max(N_c, N_v) \times (N_\mu/P_Y) \times N_k$ | $400 \times 25 \times 1000 \times 16\text{B} = 160$ MB |
| **Total** | | **~9 GB** |

Compare: old scheme with replicated X would require $n_\text{vec} \times N_v N_c N_k = 200 \times 160\text{M} \times 16\text{B} = 512$ GB per proc for Krylov vectors alone.

### Reduce-scatter temporary bounding

The RS temporary ($N_c \times N_\mu/P_Y \times N_k$ for step 4, or $N_v \times N_c/P_X \times N_k$ for step 5) can be bounded by chunking: process the band index in blocks of size $N_c/P_X$ (or $N_v/P_Y$) and do sequential reduce-scatters. Same total comm, temp bounded to one Krylov vector slice ($N_v N_c N_k / P$).

---

## 9. Ring Implementation Notes

### Ring as allgather + compute

An equivalent implementation replaces the explicit ring with:
1. `allgather` the X chunk over Y → each proc gets $X(v, c_X, \mathbf{k})$ for all $v$
2. Local GEMM to compute $A$

**Tradeoff:** allgather materializes the full gathered tensor ($P_Y$ ring buffers), requiring $P_Y$× more memory than the ring. For $P_Y = 8$: 320 MB vs 40 MB.

In JAX/XLA, `jax.lax.all_gather` with a subsequent matmul may be auto-optimized into an overlapped ring-compute pipeline. Alternatively, `jax.lax.ppermute` can implement explicit ring communication with manual compute-comm overlap.

### Fusing V step 1v with W step 1

During the W step 1 ring, each arriving $v$-chunk is used for two purposes:
1. Building $A(\nu_Y, c_X, \mathbf{k}')$ (for W)
2. Accumulating $a^\mathbf{Q}(\nu_Y)$ (for V)

The V accumulation adds negligible cost (it contracts over both $v$ and $c$ at each ν point, producing a scalar per ν). Fusing avoids a separate ring pass for V.

---

## 10. Pre-computation (Amortized Over Iterations)

1. **Pre-FFT W**: Store $W_\mathbf{R}(\mu_X, \nu_Y)$ instead of $W_\mathbf{q}$. Eliminates one FFT per $(\mu,\nu)$ pair per matvec.

2. **$\tilde{V}^\mathbf{Q}$**: Precompute for each Q needed. Cost: $O(N_k (N_\mu^V)^2 N_g)$ total for all Q, one-time. The matrix is tiny and replicated.

3. **Quasiparticle energies**: $\epsilon_{c,\mathbf{k}+\mathbf{Q}}$ and $\epsilon_{v,\mathbf{k}}$ stored locally. Shape: $N_c/P_X \times N_k$ and $N_v/P_Y \times N_k$. Negligible.

---

## 11. Krylov Iteration

### Inner products

$$\langle X, Y \rangle = \sum_{v_Y, c_X, \mathbf{k}} X^*(v_Y, c_X, \mathbf{k})\; Y(v_Y, c_X, \mathbf{k})$$

Local partial sum + **allreduce** of a single complex number over all $P$ procs. Negligible.

### Orthogonalization

Each Lanczos step requires $O(n_\text{iter})$ inner products and AXPYs on vectors of size $N_v N_c N_k / P$ per proc. Cost: $O(n_\text{iter} \cdot N_v N_c N_k / P)$ FLOPs + $O(n_\text{iter})$ scalar allreduces.

---

## 12. Extension: Beyond-TDA (Full BSE)

The full BSE includes coupling blocks $V_B, W_B$. In ISDF form:

$$W_B^\mathbf{Q}(i_v i_c \mathbf{k},\; j_v j_c \mathbf{k}') = \frac{1}{N_k}\sum_{\mu,\nu} \bar{u}_{i_c,\mathbf{k}+\mathbf{Q}}(\hat{r}_\mu)\, u_{j_v,\mathbf{k}'+\mathbf{Q}}(\hat{r}_\mu)\;\tilde{W}_{\mathbf{k}-\mathbf{k}',\mu\nu}\; \bar{u}_{j_c,\mathbf{k}'}(\hat{r}_\nu)\, u_{i_v,\mathbf{k}}(\hat{r}_\nu)$$

Same convolution structure, different band pairings on each ISDF leg (conduction with valence instead of conduction with conduction on the μ leg). The parallelization is identical with relabeled band indices in steps 2 and 4. The B-blocks double the matvec work but do not change the communication pattern.

---

## 13. Practical Recommendations

### Choosing $P_X$ vs $P_Y$

The dominant ring comm is $N_v N_c N_k / P_X$. The Krylov vector distribution is $N_v/P_Y \times N_c/P_X$. For balanced memory and comm when $N_c = N_v$, use $P_X = P_Y = \sqrt{P}$.

### When to add a third axis (K)

If $N_k \gtrsim 5000$ and wavefunction memory per proc exceeds budget, distribute k over $P_K = 2\text{--}4$. The FFT convolution then requires a distributed FFT (allgather over K + local FFT, or all-to-all transpose). Use sparingly since convolution is subdominant in the many-band regime.

### Parallelism limits

$$P \lesssim \min\Big(\frac{N_\mu^2}{4},\; N_v N_c\Big)$$

With $N_\mu = 200$ and $N_v N_c = 160\text{K}$: $P \lesssim 10000$. The band dimension is no longer the bottleneck.