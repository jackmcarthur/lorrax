Below is a **“drop-in spec”** for a **block matvec** (X \mapsto HX = DX + 2VX - WX) in the Henneke-style (D/V/W) partition, using your **Approach A** idea:

* primary parallelism on **(\mu) (axis (X))** and **(\nu) (axis (Y_\nu))**,
* the **block of trial vectors** sharded on **(Y_b)**,
* **no sharding in (\mathbf k)** (so NUFFT/FFT stays local; no pencil all-to-all).

The only “creative” trick is: even if you physically have a 2D mesh (X\times Y), you should **logically reshape (Y = Y_\nu \times Y_b)** so that reductions over (\nu) never mix different block-slices (b).

---

## 0) Logical mesh + sharding declarations

Let devices form a logical mesh
[
\mathcal M ;=; X \times Y_\nu \times Y_b,
\qquad |X|,|Y_\nu|,|Y_b| = P.
]

Use sharding annotations like:

* (\mu_X): (\mu) sharded across (X)
* (\nu_{Y_\nu}): (\nu) sharded across (Y_\nu)
* (b_{Y_b}): block index (b) sharded across (Y_b)
* indices with no annotation are replicated.

**Stored data (your scheme):**

* Screened interaction in real space (or pretransformed):
  [
  W_{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu})
  ]
* Bare exchange metric (ISDF Coulomb-like):
  [
  \widetilde V(\mu_X,\nu_{Y_\nu})
  ]
* Wavefunctions stored in two copies:
  [
  u^{(\mu)}*{n\mathbf k}(\mu_X),\qquad u^{(\nu)}*{n\mathbf k}(\nu_{Y_\nu})
  ]
  replicated over the other logical axes ((Y_\nu,Y_b) or (X,Y_b) respectively).

**Trial block and output layout (recommended):**

* Keep (\mathbf k) replicated (no pencil), keep (v) replicated, and shard (c) on (X) so you can use a **reduce-scatter(_X)** instead of an allreduce of a huge (vckb) object:
  [
  X(b_{Y_b},,v,,c_X,,\mathbf k).
  ]
  This is the only “non-(\mu,\nu)” sharding I’m recommending; it prevents an otherwise brutal allreduce when reconstructing ((WX)) / ((VX)).

**Collectives notation:**

* (\operatorname{psum}*{Y*\nu}(\cdot)): reduction over the (\nu)-sharding axis only (safe because (b) is on (Y_b)).
* (\operatorname{reduce_scatter}_{X\to c_X}(\cdot)): reduce across (X) and scatter into the (c_X) partition.

---

## 1) Diagonal term (D): local

For each block vector,
[
(DX)(b_{Y_b},v,c_X,\mathbf k)
=============================

\big(\varepsilon_{c\mathbf k}-\varepsilon_{v\mathbf k}\big),
X(b_{Y_b},v,c_X,\mathbf k).
]
**Comms:** none.

---

## 2) Screened-direct term (W): NUFFT/FFT convolution + elementwise multiply in ((\mu,\nu))

This corresponds to the paper’s regrouped screened term (their Eq. 4–6): a (\mathbf k-\mathbf k') convolution for each ((\mu,\nu)), which you implement by NUFFT/FFT to (\mathbf R), then multiply by (W_{\mathbf R}(\mathbf R,\mu,\nu)), then inverse transform.

### 2.1 Build the “pair amplitude” on the (\nu)-side (valence contraction)

Define (for each (b,c,\mathbf k,\nu)):
[
T(b_{Y_b},c_X,\mathbf k,\nu_{Y_\nu})
====================================

\sum_{v=1}^{N_v}
\overline{u^{(\nu)}*{v\mathbf k}(\nu*{Y_\nu})};
X(b_{Y_b},v,c_X,\mathbf k).
]
**Comms:** none (since (v) is replicated).

### 2.2 Build the ((\mu,\nu)) pair object (conduction contraction)

Define
[
S(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b})
======================================

\sum_{c=1}^{N_c}
u^{(\mu)}*{c\mathbf k}(\mu_X);
T(b*{Y_b},c,\mathbf k,\nu_{Y_\nu}).
]
Because (c) is sharded as (c_X), do it in two steps:

* local partial over (c_X):
  [
  S^{\text{partial}}(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b})
  =
  \sum_{c_X\in\text{local}(X)}
  u^{(\mu)}*{c\mathbf k}(\mu_X);
  T(b*{Y_b},c_X,\mathbf k,\nu_{Y_\nu}),
  ]
* then **psum over (X)** to finish the full (c)-sum:
  [
  S(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b})
  =
  \operatorname{psum}*{X}!\Big(S^{\text{partial}}(\mathbf k,\mu_X,\nu*{Y_\nu},b_{Y_b})\Big).
  ]

> This is an unavoidable reduction **if you shard (c)**. If you instead replicate (c), this psum disappears—at the cost of larger per-GPU (X) storage.

### 2.3 NUFFT/FFT (\mathbf k \to \mathbf R) (local)

[
S_{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu},b_{Y_b})
==================================================

\mathcal F_{\mathbf k\to \mathbf R}!\left[S(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b})\right].
]
**Comms:** none (because (\mathbf k) replicated; this is why I’m not recommending (\mathbf k)-sharding).

### 2.4 Screened multiply in (\mathbf R)-space (local, your “core”)

[
U_{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu},b_{Y_b})
==================================================

W_{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu})
\odot
S_{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu},b_{Y_b}).
]
**Comms:** none.

### 2.5 Inverse NUFFT/FFT (\mathbf R \to \mathbf k) (local)

[
U(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b})
======================================

\mathcal F^{-1}*{\mathbf R\to \mathbf k}!\left[U*{\mathbf R}(\mathbf R,\mu_X,\nu_{Y_\nu},b_{Y_b})\right].
]
**Comms:** none.

### 2.6 Decode back to ((v,c,\mathbf k)) without an allreduce (use reduce-scatter)

First contract over (\nu) locally:
[
M(\mathbf k,\mu_X,b_{Y_b},v)
============================

\sum_{\nu\in \nu_{Y_\nu,\text{local}}}
u^{(\nu)}*{v\mathbf k}(\nu*{Y_\nu});
U(\mathbf k,\mu_X,\nu_{Y_\nu},b_{Y_b}).
]

Then contract over (\mu_X) locally to produce *partial* ((c,\dots)):
[
Z^{\text{partial}}(b_{Y_b},v,c,\mathbf k)
=========================================

\sum_{\mu\in \mu_{X,\text{local}}}
\overline{u^{(\mu)}*{c\mathbf k}(\mu_X)};
M(\mathbf k,\mu_X,b*{Y_b},v).
]

Now combine across (X) **and scatter into (c_X)**:
[
(WX)(b_{Y_b},v,c_X,\mathbf k)
=============================

\operatorname{reduce_scatter}*{X\to c_X}!\Big(Z^{\text{partial}}(b*{Y_b},v,c,\mathbf k)\Big).
]

**Comms summary for (W):**

* (\operatorname{psum}_X) in the **encode** if you shard (c),
* (\operatorname{reduce_scatter}_{X\to c_X}) in the **decode** (this replaces a huge allreduce).

No (\nu)-axis collective is needed here because (\nu) is only summed locally within each (\nu)-shard in the decode step (you’re not trying to form an intermediate that requires the full (\nu)-sum globally; you push it into the final contraction).

---

## 3) Exchange term (V): ISDF metric multiply (\widetilde V(\mu,\nu))

This corresponds to the paper’s regrouped exchange term (their Eq. 4–5): it has **no (\mathbf k-\mathbf k') kernel**, so the only “global” piece is a plain sum over (\mathbf k') that is local if (\mathbf k) is replicated.

### 3.1 Build a (\nu)-localized “polarization-like” source (local)

Same first contraction as in (W):
[
T(b_{Y_b},c_X,\mathbf k,\nu_{Y_\nu})
====================================

\sum_{v}
\overline{u^{(\nu)}*{v\mathbf k}(\nu*{Y_\nu})};
X(b_{Y_b},v,c_X,\mathbf k).
]
**Comms:** none.

Then contract over (c) (same caveat as before):
[
P^{\text{partial}}(b_{Y_b},\mathbf k,\nu_{Y_\nu})
=================================================

\sum_{c_X\in\text{local}(X)}
u^{(\nu)}*{c\mathbf k}(\nu*{Y_\nu});
T(b_{Y_b},c_X,\mathbf k,\nu_{Y_\nu}),
]
[
P(b_{Y_b},\mathbf k,\nu_{Y_\nu})
================================

\operatorname{psum}*{X}!\Big(P^{\text{partial}}(b*{Y_b},\mathbf k,\nu_{Y_\nu})\Big).
]

Now sum over (\mathbf k) locally (since (\mathbf k) replicated):
[
p(b_{Y_b},\nu_{Y_\nu})
======================

\sum_{\mathbf k} P(b_{Y_b},\mathbf k,\nu_{Y_\nu}).
]
**Comms:** none.

### 3.2 Apply the (\widetilde V) metric (this is the only (\nu)-axis collective)

[
q^{\text{partial}}(b_{Y_b},\mu_X)
=================================

\sum_{\nu\in\nu_{Y_\nu,\text{local}}}
\widetilde V(\mu_X,\nu_{Y_\nu});
p(b_{Y_b},\nu_{Y_\nu}),
]
[
q(b_{Y_b},\mu_X)
================

\operatorname{psum}*{Y*\nu}!\Big(q^{\text{partial}}(b_{Y_b},\mu_X)\Big).
]

This (\operatorname{psum}*{Y*\nu}) is “the right” collective: it sums across the (\nu)-shards **within each block-group** (no mixing across (b)).

### 3.3 Form ((VX)) in ((v,c,\mathbf k)) with reduce-scatter on (X)

For each (\mathbf k), form the (\mu)-weighted scalar
[
s(b_{Y_b},\mathbf k,\mu_X)
==========================

\overline{u^{(\mu)}*{c\mathbf k}(\mu_X)},u^{(\mu)}*{v\mathbf k}(\mu_X); q(b_{Y_b},\mu_X)
]
(interpret the (u^{(\mu)}*{v\mathbf k}) factor according to how you map “valence on (\mu)” in your implementation; some codes use (\mu) for both legs in (V), consistent with the paper’s (u*{i_c}\bar u_{j_v}) products at the same point.)

Then contract over (\mu_X) locally to get a partial ((v,c)) output:
[
(VX)^{\text{partial}}(b_{Y_b},v,c,\mathbf k)
============================================

\sum_{\mu\in\mu_{X,\text{local}}}
\overline{u^{(\mu)}*{c\mathbf k}(\mu_X)};u^{(\mu)}*{v\mathbf k}(\mu_X);q(b_{Y_b},\mu_X).
]

Finally combine across (X) and scatter into (c_X):
[
(VX)(b_{Y_b},v,c_X,\mathbf k)
=============================

\operatorname{reduce_scatter}*{X\to c_X}!\Big((VX)^{\text{partial}}(b*{Y_b},v,c,\mathbf k)\Big).
]

**Comms summary for (V):**

* (\operatorname{psum}_X) if (c) is sharded (building (p)),
* **one** (\operatorname{psum}*{Y*\nu}) for the (\widetilde V) multiply,
* (\operatorname{reduce_scatter}_{X\to c_X}) to assemble the final ((VX)) without allreducing.

---

## 4) Assemble the full block matvec (local)

[
(HX)(b_{Y_b},v,c_X,\mathbf k)
=============================

(DX)(b_{Y_b},v,c_X,\mathbf k)
;+;2,(VX)(b_{Y_b},v,c_X,\mathbf k)
;-;(WX)(b_{Y_b},v,c_X,\mathbf k).
]
**Comms:** none.

---

## Why I did **not** shard (\mathbf k)

If you shard (\mathbf k) while insisting on no pencil FFT, your NUFFT/FFT-based convolution becomes either:

* incorrect (missing contributions), or
* requires an all-to-all / transpose-equivalent to do the transform globally.

So **(\mathbf k)-sharding is only “best” if you accept pencil comm**. Since you explicitly want to avoid that, the best practical choice is:

* keep (\mathbf k) replicated,
* shard the block (b) on (Y_b),
* shard (\mu) on (X) and (\nu) on (Y_\nu),
* shard (c) on (X) *only to enable reduce-scatter output assembly* (so you don’t allreduce the whole (vckb) tensor).

---

If you want one extra level of concreteness for the “another agent” to implement: the critical engineering detail is the **logical mesh reshape**
[
Y ;\mapsto; (Y_\nu, Y_b)
]
so that every (\operatorname{psum}*{Y*\nu}) is performed *within a fixed (b)-group*. In JAX terms that just means you define the mesh with three axis names, even if physically it’s a 2D slab, and then use `PartitionSpec(mu='X', nu='Ynu', b='Yb', ...)` everywhere.
