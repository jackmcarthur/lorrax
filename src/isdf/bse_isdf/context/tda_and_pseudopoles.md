Below is a single, implementation-first framework that (i) upgrades your existing **TDA Casida matvec** (built from **D, V, W** pieces) to the **full non-TDA** block matvec used by GMRES/FEAST, and (ii) builds **windowed pseudopoles** ({(\Omega_p^{(\mathbf q)}, d_p^{(\mathbf q)}[r_\mu])}) such that
[
W_c^{(+)}(\omega;\mathbf q);\approx;\sum_p \frac{d_p^{(\mathbf q)} d_p^{(\mathbf q),\dagger}}{\omega-\Omega_p^{(\mathbf q)}}.
]

I’ll use bracket notation like `R^(q)[cvk, r_mu]`, and I’ll be explicit about (\mathbf k-\mathbf q).

---

## 0) Indices and core tensors

### Transition index (fixed (\mathbf q))

Let

* (t \equiv (v,c,\mathbf k)) label the transition (|v,\mathbf k\rangle \to |c,\mathbf k-\mathbf q\rangle).
* Transition vectors are `X[t] = X[v,c,k]`, `Y[t] = Y[v,c,k]`.

### ISDF / auxiliary index

* (\mu) indexes ISDF points (r_\mu). Density-space vectors are `rho[r_mu]`.

### Transition→density map (R^{(\mathbf q)})

Define
[
R^{(\mathbf q)}[t, r_\mu] \equiv \psi^**{c,\mathbf k-\mathbf q}(r*\mu),\psi_{v,\mathbf k}(r_\mu).
]
Then for any transition vector `Z[t]`,
[
(R^{(\mathbf q)} Z)[r_\mu] = \sum_{t} R^{(\mathbf q)}[t,r_\mu]; Z[t].
]
Adjoint (density→transition):
[
(R^{(\mathbf q)\dagger} \eta)[t] = \sum_{\mu} R^{(\mathbf q)}[t,r_\mu]^*; \eta[r_\mu].
]

### Coulomb metric (v^{(\mathbf q)})

`v^(q)[r_mu, r_nu]` acts in density space:
[
(v^{(\mathbf q)} \rho)[r_\mu]=\sum_{\nu} v^{(\mathbf q)}[r_\mu,r_\nu];\rho[r_\nu].
]

---

## 1) Non-TDA block operator you will FEAST-filter

You will **not** run FEAST/GMRES on the generalized (\mathcal H\Phi=\Omega J\Phi). You will fold (J) in and use the standard eigenproblem
[
S^{(\mathbf q)} \Phi = \Omega \Phi,
\qquad
\Phi \equiv \begin{pmatrix}X\Y\end{pmatrix},
]
with block structure
[
S^{(\mathbf q)} ;=;
\begin{pmatrix}
A^{(\mathbf q)} & B^{(\mathbf q)} \
-,B^{(\mathbf q)\dagger} & -,A^{(\mathbf q)\dagger}
\end{pmatrix}.
]

You said your kernels naturally come in three pieces:

* **D**: diagonal energy differences,
* **V**: bare Coulomb “exchange/Hartree-like” piece,
* **W**: screened direct piece (static (W(\omega=0))).

So define (what your code should implement as black-box matvecs):
[
A = D + V - W,\qquad B = V - W,
]
with the understanding:

* **RPA** corresponds to setting (W\equiv 0) and using the RPA/Hartree form of (V).
* **Static BSE** uses both (V) (bare exchange, with spin prefactor) and (W) (screened direct at (\omega=0)).

### 1.1 Full matvec (what GMRES/FEAST calls)

Given input `(X[t], Y[t])`, return `(Xout[t], Yout[t])`:
[
\begin{aligned}
X_{\rm out} &= (A X) + (B Y),\
Y_{\rm out} &= -,(B^\dagger X) - (A^\dagger Y).
\end{aligned}
]
So implementation-wise you need routines:

* `AX = apply_A(X)` and `BY = apply_B(Y)`,
* `BHX = apply_BH(X)` and `AHY = apply_AH(Y)`.

If your operators are real-symmetric in the chosen gauge you can set adjoint=transpose; otherwise implement conjugate-adjoint.

---

## 2) Explicit ISDF contractions for the **V** and **W** pieces (non-TDA)

This section tells you exactly what “apply_V” and “apply_W” mean for both (A) and (B), in terms of wavefunctions and ISDF.

### 2.1 The D piece (same as TDA)

[
(D Z)[v,c,\mathbf k] \equiv \Delta[v,c,\mathbf k]; Z[v,c,\mathbf k],
\quad
\Delta = \varepsilon_{c,\mathbf k-\mathbf q}-\varepsilon_{v,\mathbf k}.
]

### 2.2 RPA/Hartree-style (V) (depends only on (\mathbf q), uses transition densities)

This is the “three-step (R)-(v)-(R^\dagger)” contraction.

Define for any transition vector `Z[t]` two density contractions:

* **resonant density**: `rho = R Z`
  [
  \rho[r_\nu] = \sum_{t} R^{(\mathbf q)}[t,r_\nu];Z[t]
  ]
* **antiresonant density**: `rho = conj(R) Z`
  [
  \rho[r_\nu] = \sum_{t} R^{(\mathbf q)}[t,r_\nu]^*;Z[t]
  ]

Then apply Coulomb:
[
\phi[r_\mu] = \sum_{\nu} v^{(\mathbf q)}[r_\mu,r_\nu];\rho[r_\nu].
]

Back-project:
[
(V_{\rm RPA} Z)[t] = \sum_{\mu} R^{(\mathbf q)}[t,r_\mu]^*;\phi[r_\mu].
]

Now use this in the blocks as:

* **A-side (V) acting on (X)**: use `rho = R X`.
* **B-side (V) acting on (Y)**: use `rho = conj(R) Y`.

This is the standard non-TDA Casida permutation in your (R=\psi_c^*\psi_v) convention: the (B) block sees the deexcitation density.

(For static BSE exchange, the same *topology* applies; it just carries the spin factor and sign convention you already use in TDA.)

### 2.3 Static screened direct (W) (expensive, momentum transfer (\mathbf k-\mathbf k'))

This is the “rotated topology” term: electron–electron at one vertex, hole–hole at the other, coupled by screened (W(\mathbf Q,0)) with (\mathbf Q=\mathbf k-\mathbf k').

Let (W^{(\mathbf Q)}[r_\mu,r_\nu]\equiv W(\mathbf Q;\omega=0)) be your static screened operator in the same ISDF basis, with a fast batched apply on (\mu) for each fixed (\nu).

#### W contribution to (A X)

For each (\mathbf k'), form the mixed density intermediate (size (\mu\times\nu)):
[
M^X_{\mu\nu}(\mathbf k') =
\sum_{v',c'}
\psi_{c',\mathbf k'-\mathbf q}(r_\mu);
X[v',c',\mathbf k'];
\psi^**{v',\mathbf k'}(r*\nu).
]
Then apply screened interaction on the (\mu) index:
[
\widetilde M^X_{\mu\nu}(\mathbf k',\mathbf k)
=============================================

\sum_{\mu'} W^{(\mathbf k-\mathbf k')}[r_\mu,r_{\mu'}];M^X_{\mu'\nu}(\mathbf k').
]
Finally contract onto the output transition ((v,c,\mathbf k)):
[
(W_A X)[v,c,\mathbf k]
======================

\sum_{\mathbf k'}\sum_{\mu\nu}
\psi^**{c,\mathbf k-\mathbf q}(r*\mu);
\widetilde M^X_{\mu\nu}(\mathbf k',\mathbf k);
\psi_{v,\mathbf k}(r_\nu).
]
Your overall sign convention (attractive) is handled in the definition (A = D + V - W).

#### W contribution to (B Y)

Use the swapped mixed density appropriate to the deexcitation sector:
[
M^Y_{\mu\nu}(\mathbf k') =
\sum_{v',c'}
\psi_{v',\mathbf k'}(r_\mu);
Y[v',c',\mathbf k'];
\psi^**{c',\mathbf k'-\mathbf q}(r*\nu).
]
Then the same screened apply:
[
\widetilde M^Y_{\mu\nu}(\mathbf k',\mathbf k)
=============================================

\sum_{\mu'} W^{(\mathbf k-\mathbf k')}[r_\mu,r_{\mu'}];M^Y_{\mu'\nu}(\mathbf k'),
]
and the same final contraction:
[
(W_B Y)[v,c,\mathbf k]
======================

\sum_{\mathbf k'}\sum_{\mu\nu}
\psi^**{c,\mathbf k-\mathbf q}(r*\mu);
\widetilde M^Y_{\mu\nu}(\mathbf k',\mathbf k);
\psi_{v,\mathbf k}(r_\nu).
]

Then implement:
[
(A X)= (D X) + (V_A X) - (W_A X),\qquad
(B Y)= (V_B Y) - (W_B Y),
]
with (V_A,V_B) defined by the RPA/exchange three-step contraction above (including your singlet/triplet prefactor if applicable).

---

## 3) What FEAST filtering returns in the non-TDA case

FEAST filtering uses shifted solves of (S):
[
(z I - S),\Psi = \Phi,\quad \Phi=\binom{X}{Y}.
]
You already have FEAST; all you need is the ability to solve with this **augmented** operator (S) (dimension (2N_{\rm trans})).

Windows are specified as:
[
[\Omega_{w,\min},,\Omega_{w,\max}].
]
Your FEAST projector onto that window is realized by a quadrature:
[
P_w(S),\Phi \approx \sum_{\ell=1}^{n_{\rm quad}} w_{w,\ell},(z_{w,\ell}I-S)^{-1}\Phi.
]

---

## 4) Per-window pseudopole construction (non-TDA, pole+rank-1 residue output)

### Fixed per-window integers

Use consistent names:

* `m0` = number of random seeds per window
* `nquad` = contour quadrature points
* `p_keep` = number of bright modes retained
* `n_tail` = number of stochastic tail pseudomodes

### Step A — Density-biased seeding into both (X) and (Y)

For (j=1..m0):

1. Draw random density vector `eta_j[r_mu]`.
2. `tmp[r_mu] = sum_nu v^(q)[r_mu,r_nu] * eta_j[r_nu]`.
3. `f_j[cvk] = sum_mu conj(R^(q)[cvk,r_mu]) * tmp[r_mu]`  (this is (R^\dagger v,\eta)).
4. Set the augmented seed
   [
   \Phi^{(j)}_0 = \binom{X_0}{Y_0} = \binom{f_j}{-\,\bar f_j},
   ]
   where the antiresonant drive uses the transpose-coupled vertex (\bar f=d^T v\eta) for complex Bloch spinors.
   This targets modes with large density coupling (large (R X + R^* Y)), which is exactly what feeds (W).

### Step B — FEAST filter into the window

For each (j), compute filtered vector:
[
\tilde\Phi^{(j)} = P_w(S),\Phi^{(j)}*0
\approx \sum*{\ell=1}^{nquad} w_{w,\ell};\Psi^{(j)}*{w,\ell},
\quad (z*{w,\ell}I-S)\Psi^{(j)}_{w,\ell}=\Phi^{(j)}*0.
]
Stack columns:
[
\tilde\Phi_w \in \mathbb C^{(2N*{\rm trans})\times m0}.
]

### Step C — Orthonormalize and build reduced operator

Compute an orthonormal basis:
[
V_w = \mathrm{orth}(\tilde\Phi_w)\in \mathbb C^{(2N_{\rm trans})\times m_w}.
]
Form reduced matrix:
[
H_w = V_w^\dagger S V_w \in \mathbb C^{m_w\times m_w}.
]
(Compute `SV_w` by applying your augmented matvec (S) to each basis column.)

### Step D — Build the “residue snapshot” matrix once (uses the density channel)

Split each basis column (V_w[:,j]) into transition blocks:
[
V_w[:,j] \equiv \binom{X^{(j)}}{Y^{(j)}}.
]
Form the residue snapshot column in density space:
[
d^{(j)}[r_\mu]
==============

\sum_{\nu} v^{(\mathbf q)}[r_\mu,r_\nu];
\Big(\sum_{t} R^{(\mathbf q)}[t,r_\nu]\;X^{(j)}[t]\;+\;\sum_{t} R^{(\mathbf q)}[t,r_\nu]^*\;Y^{(j)}[t]\Big).
]
As a matrix, define
[
C_w[r_\mu, j] \equiv d^{(j)}[r_\mu] \in \mathbb C^{N_\mu\times m_w}.
]
This is the key caching step: after you have `C_w`, **every residue vector you output is just `C_w @ g` for some small coefficient vector `g`**.

### Step E — Brightness eigen-decomposition in the reduced subspace

Compute the small Gram matrix:
[
G_w = C_w^\dagger C_w \in \mathbb C^{m_w\times m_w}.
]
Diagonalize:
[
G_w = W_w,\mathrm{diag}(\sigma_1^2,\dots,\sigma_{m_w}^2),W_w^\dagger,
\quad \sigma_1\ge\sigma_2\ge\cdots.
]
Let `Wb = W_w[:, 1:p_keep]` and `Wd = W_w[:, p_keep+1:m_w]`.

### Step F — Bright pseudopoles (distinct poles via Ritz in bright subspace)

Form reduced bright operator:
[
H_b = W_b^\dagger H_w W_b \in \mathbb C^{p_{\rm keep}\times p_{\rm keep}}.
]
Diagonalize:
[
H_b u_j = \Omega_{w,j},u_j,\qquad j=1..p_{\rm keep}.
]
For each (j), form coefficients in the (V_w) basis:
[
g_{w,j} = W_b u_j \in \mathbb C^{m_w},
]
and output residue vector:
[
d_{w,j} = C_w g_{w,j} \in \mathbb C^{N_\mu}.
]
Store the pair ((\Omega_{w,j}, d_{w,j})).

### Step G — Stochastic tail pseudopoles (few additional rank-1 terms)

Define dim reduced operator:
[
H_d = W_d^\dagger H_w W_d \in \mathbb C^{m_d\times m_d},
\quad m_d=m_w-p_{\rm keep}.
]
For (\ell=1..n_{\rm tail}):

1. draw random normalized `z_ell ∈ C^{m_d}`.
2. assign a pole by Rayleigh quotient:
   [
   \Omega_{w,\ell}^{\rm tail} = z_\ell^\dagger H_d z_\ell.
   ]
3. coefficient vector in the (V_w) basis:
   [
   g_{w,\ell}^{\rm tail} = W_d z_\ell \in \mathbb C^{m_w}.
   ]
4. residue vector:
   [
   d_{w,\ell}^{\rm tail} = C_w g_{w,\ell}^{\rm tail}.
   ]

Enforce total discarded brightness match (single mandatory rescale):

* Discarded brightness: (B_{\rm disc}=\sum_{i=p_{\rm keep}+1}^{m_w}\sigma_i^2).
* Tail brightness: (B_{\rm tail}=\sum_{\ell=1}^{n_{\rm tail}}|d_{w,\ell}^{\rm tail}|^2).
* Rescale all tail residues by (\alpha=\sqrt{B_{\rm disc}/B_{\rm tail}}):
  [
  d_{w,\ell}^{\rm tail} \leftarrow \alpha, d_{w,\ell}^{\rm tail}.
  ]
  Store (\big(\Omega_{w,\ell}^{\rm tail}, d_{w,\ell}^{\rm tail}\big)).

---

## 5) Final output and how it reproduces (W(\omega))

Collect all pairs from all windows (w):

* bright: ((\Omega_{w,j}, d_{w,j})), (j=1..p_{\rm keep})
* tail: ((\Omega_{w,\ell}^{\rm tail}, d_{w,\ell}^{\rm tail})), (\ell=1..n_{\rm tail})

Then the resonant screened interaction approximation is
[
W_c^{(+)}(\omega;\mathbf q) \approx \sum_{w}\left[
\sum_{j=1}^{p_{\rm keep}}\frac{d_{w,j}d_{w,j}^\dagger}{\omega-\Omega_{w,j}}
;+;
\sum_{\ell=1}^{n_{\rm tail}}\frac{d_{w,\ell}^{\rm tail}d_{w,\ell}^{\rm tail\dagger}}{\omega-\Omega_{w,\ell}^{\rm tail}}
\right].
]

The crucial non-TDA detail is that **every residue is built from (R X + R^* Y)** through `C_w`, i.e. from the density channel that actually screens.

---

If you want, I can now convert this directly into a “call graph” for your codebase (`apply_S_full`, `apply_V_A`, `apply_V_B`, `apply_W_A`, `apply_W_B`, `feast_filter_window`, `build_Cw`, etc.) with explicit array shapes like `X[v,c,k]`, `psi_c_kmq[mu]`, and batching points for GPU.
