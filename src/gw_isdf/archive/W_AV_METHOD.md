Here’s a compact, “methods-section” recipe you can drop into your codebase. I’ve split it into: (A) target quantity and averaging; (B) ISDF objects and small-q model; (C) stable construction of the wing tensor with projector-regularized (k!\cdot!p); and (D) an efficient implementation plan (what to precompute, what to multiply, and in what order).

---

# A) Target: Voronoi-cell average of the head (W_{00}(\mathbf q,\omega))

We want the (q!=!0)–cell average of the **head** of the screened interaction:
[
\big\langle W_{00}(\omega)\big\rangle_{\text{Vor}} ;=; \frac{1}{\Omega_{\text{Vor}}}\int_{\text{Vor}(0)}!!d^d q;
W_{00}(\mathbf q,\omega+i\eta),
]
evaluated using a small-(\mathbf q) model of the **irreducible polarizability head**
[
\chi^0_{00}(\mathbf q,\omega+i\eta) ;\equiv; f(\mathbf q,\omega+i\eta)
;=; \mathbf q^\top S^{\text{eff}}(\omega+i\eta),\mathbf q ;+; O(|\mathbf q|^3).
]
Then
[
\boxed{~
W_{00}(\mathbf q,\omega+i\eta) ;=; \frac{v_0(\mathbf q)}{1 - v_0(\mathbf q),f(\mathbf q,\omega+i\eta)},,\qquad
v_0^{3\text{D}}(\mathbf q)=\frac{4\pi}{|\mathbf q|^2}\ \text{or your slab}\ v_0^{2\text{D}}(\mathbf q).
~}
]
We will construct (S^{\text{eff}}) either as the **head-only** tensor (S) or the **wing-renormalized** tensor (S+ \mathrm{Re}[L_\alpha M L_\beta^\dagger]). Monte-Carlo average (W_{00}) over random (\mathbf q) in the Voronoi cell.

---

# B) ISDF objects and small-q tensors

## B.1 ISDF head/wing projectors and Coulomb blocks

* ISDF interpolation vectors: ({\zeta_\mu}*{\mu=1}^{\mu*{\max}}) at (q{=}0).
* Define the **head direction** from the physical G{=}0 component of the ISDF vectors:
  [
  u_\mu ;\equiv; \zeta_{\mu}(G{=}0)\quad\text{(unnormalized)}.
  ]
* Projectors in the ISDF index space (no ad hoc normalization):
  [
  P \equiv \frac{u\,u^\dagger}{u^\dagger u}\,,\qquad Q \equiv I - P.
  ]
* Coulomb matrix in ISDF space at (q{=}0): (V \equiv [V_{\mu\nu}] = \langle \zeta_\mu|,v,|\zeta_\nu\rangle).

  * **Head-zeroed wing block** (used for Schur dressing):
    [
    \boxed{~V_Q ;\equiv; Q^\dagger V, Q \quad\text{(i.e., rows/cols touching the head are zeroed).}~}
    ]

## B.2 Pair-samples at ISDF points and velocities

For a fixed (\mathbf k):

* Left table (L\in\mathbb C^{\mu\times N_v}), (L_{\mu v}=\psi_{v\mathbf k}^*(r_\mu)).
* Right table (R\in\mathbb C^{\mu\times N_c}), (R_{\mu c}=\psi_{c\mathbf k}(r_\mu)).
* Band gaps (\Delta_{cv}(\mathbf k)=E_{c\mathbf k}-E_{v\mathbf k}).
* Velocity matrices (v^\alpha_{mn}(\mathbf k)=\langle m\mathbf k|\hat v_\alpha|n\mathbf k\rangle).

Weights (BerkeleyGW-consistent **retarded** kernel with broadening (\eta)):
[
\boxed{~
K(\omega+i\eta;\Delta)=\frac{2\Delta}{(\omega+i\eta)^2-\Delta^2}.
~}
]

## B.3 Head-only small-q tensor (S_{\alpha\beta})

Using (|\rho(G{=}0)|^2 \approx (\mathbf q!\cdot!\mathbf v_{vc})(\mathbf q!\cdot!\mathbf v_{cv})/\Delta^2),
[
\boxed{~
S_{\alpha\beta}(\omega+i\eta)
=\frac{2}{N_k\Omega}\sum_{v c \mathbf k}(f_{v}-f_{c});
\frac{v^\alpha_{vc}(\mathbf k),v^\beta_{cv}(\mathbf k)}
{\Delta_{cv}(\mathbf k)\left[(\omega+i\eta)^2-\Delta_{cv}(\mathbf k)^2\right]}.
~}
]
(Imaginary-axis variant: replace the denominator by (-[\xi^2+\Delta^2]).)

---

# C) Wing-renormalized tensor (S^{\text{eff}}): (k!\cdot!p) + Schur complement

We need the **head–wing linear** coefficient (L_\alpha) and the **wing–wing** block (C) at (q{=}0). Everything is constructed directly in the ISDF space; no FFTs, no plane-wave (\rho_{cv}(G)) required.

## C.1 Stable (k!\cdot!p) derivatives at the ISDF points (no small-denominator blowups)

Write the (k)-derivatives of Bloch states with **projected resolvents** to avoid small denominators (near accidental degeneracy):
[
\partial_{k_\alpha}\psi_{n\mathbf k}
;=;
\sum_{m\notin\mathcal P_n}
\psi_{m\mathbf k},\frac{v^\alpha_{mn}(\mathbf k)}{i,[E_{n\mathbf k}-E_{m\mathbf k}]}
;;\longrightarrow;;
\sum_{m\notin\mathcal P_n}
\psi_{m\mathbf k},\frac{v^\alpha_{mn}(\mathbf k)}{i,[E_{n\mathbf k}-E_{m\mathbf k}]+\gamma},
]
where:

* (\mathcal P_n) is a **projected subspace** that excludes (i) the target band (n), (ii) any band within a small **safety window** (|E_m{-}E_n|<\varepsilon_{\text{kp}}) (handle those by explicit two-level mixing if needed), and (iii) respects your chosen **valence/conduction** windows;
* (\gamma) is a tiny **Tikhonov** stabilizer (e.g. (10^{-3})–(10^{-2}) eV) used only when (|E_n{-}E_m|) falls below (\varepsilon_{\text{kp}}).

At ISDF points:
[
\partial_{k_\alpha}\Phi_{vc}(\mu)
=\psi_{v}^*(r_\mu),\partial_{k_\alpha}\psi_{c}(r_\mu)
+\big(\partial_{k_\alpha}\psi_{v}(r_\mu)\big)^*,\psi_{c}(r_\mu),
]
with the **regularized** sums above for the derivatives.

Efficient construction (two GEMMs per (\alpha)):
[
D^{(\alpha)} \equiv \Psi,B^{(\alpha)}\in\mathbb C^{\mu\times N_c},\quad
B^{(\alpha)}*{nc}=\frac{v^\alpha*{nc}}{i,[E_{c}-E_{n}]+\gamma}\ \ (n\notin\mathcal P_c);
]
[
E^{(\alpha)} \equiv \Psi^* A^{(\alpha)}\in\mathbb C^{\mu\times N_v},\quad
A^{(\alpha)}*{mv}=\frac{-,v^\alpha*{vm}}{i,[E_{m}-E_{v}]+\gamma}\ \ (m\notin\mathcal P_v),
]
where (\Psi_{\mu n}=\psi_{n\mathbf k}(r_\mu)) over your working band set.

Then, elementwise over (\mu):
[
\boxed{~
\partial_{k_\alpha}\Phi_{vc}
= L_{:v}\odot D^{(\alpha)}*{:c};+;E^{(\alpha)}*{:v}\odot R_{:c},.
~}
]

## C.2 Head–wing linear coefficient (L_\alpha(\omega+i\eta))

Define the pair weights (same kernel as (S), but **no extra (1/\Delta)** here because we take the linear term on the head side):
[
C^\alpha_{vc}(\omega+i\eta) ;\equiv; \frac{2}{N_k\Omega},(f_v-f_c);
\frac{v^\alpha_{vc}(\mathbf k)}{(\omega+i\eta)^2-\Delta_{cv}(\mathbf k)^2}.
]
Accumulate a single ISDF vector (y_\alpha\in\mathbb C^{\mu}) **without** forming any (N_v\times N_c\times \mu) tensor:
[
\boxed{
\begin{aligned}
y_\alpha
&\equiv \sum_{vc} C^\alpha_{vc},\partial_{k_\alpha}\Phi_{vc}
\
&=\underbrace{\Big(L,C^\alpha\Big) \odot D^{(\alpha)}}*{\text{Hadamard over rows}}\mathbf 1_c
;;+;;
\underbrace{\Big(R,{C^\alpha}^\top\Big) \odot E^{(\alpha)}}*{\text{Hadamard over rows}}\mathbf 1_v,
\end{aligned}}
]
where (\mathbf 1_{c}) and (\mathbf 1_v) denote column-sums over (c) or (v).
Finally project to wings:
[
\boxed{~L_\alpha(\omega+i\eta) ;=; \big(Q^\dagger y_\alpha\big)^\dagger ;=; y_\alpha^\dagger ;-; (\hat u^\dagger y_\alpha),\hat u^\dagger.~}
]

## C.3 Wing–wing block (C(\omega+i\eta)) at (q{=}0)

You only need (\chi^0(\mathbf q{=}0,\omega)) **in ISDF** (which you already compute by collocation of pair products):
[
\chi^0_{\mu\nu}(\mathbf 0,\omega+i\eta)
=\frac{2}{N_k\Omega}\sum_{vc\mathbf k}(f_v-f_c);K(\omega+i\eta;\Delta_{cv});
\Phi_{vc}(\mu),\Phi_{vc}(\nu)^*.
]
Project to wings:
[
\boxed{~C(\omega+i\eta) ;\equiv; Q^\dagger,\chi^0(\mathbf 0,\omega+i\eta),Q.~}
]

## C.4 Wing screen (M(\omega+i\eta)) and the effective tensor

Solve the small linear system in ISDF-wing space:
[
\boxed{~M(\omega+i\eta) ;=; \big[I - V_Q,C(\omega+i\eta)\big]^{-1},V_Q.~}
]
Then assemble
[
\boxed{~
S^{\text{eff}}*{\alpha\beta}(\omega+i\eta)
= S*{\alpha\beta}(\omega+i\eta)

* \mathrm{Re}!\left[,L_\alpha(\omega+i\eta),M(\omega+i\eta),L_\beta(\omega+i\eta)^\dagger,\right].
  ~}
  ]
  (Force symmetry by (\tfrac12(S^{\text{eff}}+S^{\text{eff}\top})) and PSD-clip tiny negative eigenvalues for stability.)

---

# D) Efficient algorithm (what to compute, and in what order)

Below is for one frequency (\omega) (vectorize over (\omega) as you like). All GEMMs on GPU.

**Inputs (per (\mathbf k))**: (L(\mu\times N_v)), (R(\mu\times N_c)), (\Psi(\mu\times N_{\text{win}})), (v^\alpha_{mn}), (E_n), occupations, (u), (V).

**Precompute once (q=0 objects)**

1. (\hat u = u/|u|), (P=\hat u\hat u^\dagger), (Q=I-P).
2. (V_Q = Q^\dagger V Q).
3. (\chi^0(\mathbf 0,\omega+i\eta)) by your usual ISDF collocation; (C=Q^\dagger \chi^0 Q).
4. (Optional) cache (L^\dagger(\mathrm{diag}(\hat u^*),R)) if you need any head scalars later.

**Head-only tensor (S) (cheap)**
5) For each (\mathbf k): accumulate
[
S_{\alpha\beta} \mathrel{+}= \frac{2}{N_k\Omega}\sum_{vc}(f_v-f_c),
\frac{v^\alpha_{vc}v^\beta_{cv}}{\Delta_{cv}\big[(\omega+i\eta)^2-\Delta_{cv}^2\big]}.
]

**Wing ingredients (per (\alpha))**
6) Build regularized (k!\cdot!p) blocks (two GEMMs):
[
D^{(\alpha)}=\Psi,B^{(\alpha)},\quad B^{(\alpha)}*{nc}=\frac{v^\alpha*{nc}}{i(E_c-E_n)+\gamma};
]
[
E^{(\alpha)}=\Psi^* A^{(\alpha)},\quad A^{(\alpha)}*{mv}=\frac{-,v^\alpha*{vm}}{i(E_m-E_v)+\gamma},
]
with band-exclusion by (\mathcal P_n) (skip rows/cols where (|E_m{-}E_n|<\varepsilon_{\text{kp}}) and, if needed, add explicit two-level mixing for those).

7. Form pair weights (no extra (1/\Delta)):
   [
   C^\alpha_{vc}=\frac{2}{N_k\Omega}(f_v-f_c),\frac{v^\alpha_{vc}}{(\omega+i\eta)^2-\Delta_{cv}^2}.
   ]
8. Two GEMMs for the accumulators:
   [
   G^\alpha_c \equiv L,C^\alpha \in \mathbb C^{\mu\times N_c},\qquad
   H^\alpha_v \equiv R,{C^\alpha}^\top \in \mathbb C^{\mu\times N_v}.
   ]
9. Row-wise Hadamards and column-sums to (\mu)-vector:
   [
   y_\alpha = \big(G^\alpha_c \odot D^{(\alpha)}\big),\mathbf 1_c
   + \big(H^\alpha_v \odot E^{(\alpha)}\big),\mathbf 1_v.
   ]
10. Project to wings: (L_\alpha = (Q^\dagger y_\alpha)^\dagger = y_\alpha^\dagger - (\hat u^\dagger y_\alpha),\hat u^\dagger).

**Wing screen & effective tensor**
11) Solve (M=(I - V_Q C)^{-1} V_Q) (ISDF-wing dimension; tiny, reuse across (\alpha,\beta)).
12) Build (S^{\text{eff}} = S + \mathrm{Re}[L_\alpha M L_\beta^\dagger]) (form all (\alpha,\beta\in{x,y,(z)})).

**MC average**
13) Sample (\mathbf q_j) in the Voronoi cell ((j=1\ldots M), (M\sim 10^5!-!10^6)):
- (f_j = \mathbf q_j^\top S^{\text{eff}},\mathbf q_j).
- (W_{00}(\mathbf q_j)=v_0(\mathbf q_j)/\big[1 - v_0(\mathbf q_j) f_j\big]).
14) Average (\langle W_{00}\rangle \approx \frac{1}{M}\sum_j W_{00}(\mathbf q_j)).

**(Optional) Head patch in the ISDF matrix**
15) If you want ISDF-matrix (W_{q=0}) to carry this averaged head exactly, apply the **rank-1** update
[
W \leftarrow W + \alpha(\omega)\,u u^\dagger,\quad
\alpha(\omega)=\frac{\big\langle W_{00}(\omega)\big\rangle_{\text{Vor}} - u^\dagger W u}{\big(u^\dagger u\big)^2}.
]

Practical note: if you construct the truncated Coulomb so that the G{=}0 plane‑wave component is explicitly zeroed when forming v_q(G), then the contractions \(\sum_G \zeta^*_{\mu}(G)\,v_q(G)\,\zeta_{\nu}(G)\) already exclude the head and you do not need to explicitly project off wing components of v.

---

## Notes on stability and cost

* The only big GEMMs are steps **6** (two GEMMs per (\alpha)) and **8** (two GEMMs per (\alpha)). Everything else is (\mu)-scale (cheap).
* (k!\cdot!p) regularization: choose (\varepsilon_{\text{kp}}\sim) a few meV–10 meV; (\gamma\ll\eta) (e.g. (\gamma=10^{-3})–(10^{-2}) eV). This avoids spurious spikes in (L_\alpha) when bands come close.
* Consistency: use the **same kernel** (K(\omega+i\eta;\Delta)) in (S), (C), and (via (C^\alpha)) in (L_\alpha).
* If anisotropy is mild and you want a turbo path first: skip wings ((L_\alpha=0)) and use just (S). You can toggle wings on later; the MC/patch machinery is identical.

That’s the whole pipeline: definitions, stable (k!\cdot!p) wings, and an efficient contraction plan that never materializes (N_v!\times!N_c!\times!\mu) tensors yet yields a q-independent (S^{\text{eff}}(\omega+i\eta)) for fast (W_{00}) averaging.
