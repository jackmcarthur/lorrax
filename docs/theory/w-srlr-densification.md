# Long-range W densification (SR/LR)

`kgrid_fine` evaluates the static W on a fine q grid from its coarse-grid
tiles by trigonometric interpolation (the per-centroid-pair Wigner–Seitz
embedding in R, `bse.bse_densify.make_w_densifier`). The interpolant is exact on the coarse
points and accurate where W is analytic in q. Near Γ it is not: the screened
head goes as $v/\varepsilon$ ($1/q^2$ in bulk), and the slab-truncated kernel
$v(\mathbf K)=\frac{8\pi}{K^2}\big(1-e^{-z_c K_\parallel}\cos K_z z_c\big)$ is
non-analytic in $|q_\parallel|$ in every out-of-plane channel
$\mathbf K=\mathbf q+\mathbf G_{ws}+n\mathbf b_3$, with slope
$8\pi z_c(-1)^n/G_z^2$ at $q\to0$. `w_head_densify = c1` re-attaches only the
Γ head. `w_head_densify = srlr` moves the whole long-range part of W into a
fitted model and interpolates the remainder.

## Channels {#channels}

The long-range channels $L(q)$ are plane waves $e^{i(\mathbf q+\mathbf G_a)\cdot\mathbf r}$
(`vcoul.lr_channels`): in bulk the Wigner–Seitz slot
$\mathbf G_{ws}(q)=\arg\min_G|\mathbf q+\mathbf G|$; in a slab that slot with
$G_z=n\,b_3$, $|n|\le n_z$, where $n_z$ is the long-range criterion of
`bse.vq_interp.lr_fit_degrees` ([decisions](../architecture/decisions.md)).
The kernel is diagonal on them, $v=V_b+F^\dagger\,\mathrm{diag}(v_a)\,F$ with
$F_{a\mu}=\zeta_q(\mu,\mathbf G_a)/\sqrt\Omega$, so the remainder $V_b$ stays
positive.

## Exact split at every coarse q {#split}

With $W=(1-V\chi)^{-1}V$ (`gw.lr_channels`, beside the static Dyson solve):
$$W=J+\tilde F^\dagger D\tilde F,\qquad
J=(1-V_b\chi)^{-1}V_b,\quad
\tilde F=F(1-\chi V_b)^{-1},\quad
P=\tilde F\chi F^\dagger,\quad
D=\big[\mathrm{diag}(1/v_a)-P\big]^{-1}.$$
$\tilde F$ are the short-range-dressed form factors of the channels and $P$
their dressed proper polarizability; for one channel $D=v/(1-vP)$ is the
screened head. GW never forms $J$: with $A^{-1}=(1-\chi V)^{-1}=1+\chi W$,
$$G=F+F\chi W,\quad Q=G\,\chi F^\dagger,\quad
\tilde F=(1+Q\,\mathrm{diag}\,v)^{-1}G,\quad P=(1+Q\,\mathrm{diag}\,v)^{-1}Q,$$
three products with $n_L$ rows. The restart stores $\tilde F$ and $P$ per
irreducible q (group `W0_lr`). $\tilde F$ rows transform as ζ rows (one leg,
with the channel's G and τ phase $c_a$); $P\to c_a c_b^* P$, conjugated on
antiunitary rows.

## Model and assembly {#model}

On the coarse q with $0<|q|<r_b$ the model fits
$\tilde F_{a\mu}(q)\,e^{i\mathbf q\cdot\mathbf r_\mu}$ by a polynomial in
Cartesian q (degree ≤ 4, at most half as many monomials as points) and
$P_{ab}(q)=|q|^{m_{ab}}y_{ab}(q^2)$, with $m=2$ on the literal $G=0$ row and
column (charge neutrality) and one least-squares anchor row
$y_{00}(0)=\bar S$ from the head tensor. With
$\mathrm{LR}_m=\tilde F_m^\dagger[\mathrm{diag}(1/v)-P_m]^{-1}\tilde F_m$, $v$
exact at every q, and the window $w(|q|)$ (1 below $r_a=q_{ins}/3$, quintic
step to 0 at $r_b=0.9\,q_{ins}$, $q_{ins}$ the BZ inscribed radius):
$$X(q_c)=W(q_c)-w\,\mathrm{LR}_m(q_c),\qquad X(0)=W(0)-\mathrm{LR}(0),$$
$$W(q_f)=T[X](q_f)+w\,\mathrm{LR}_m(q_f),\qquad W(\Gamma)=T[X](0)+\mathrm{LR}(0),$$
$T$ the trigonometric densifier and $\mathrm{LR}(0)$ the exact coarse-Γ term
(the $G=0$ channel has $v=0$ there; the Γ head stays c1's). Subtracting the
model, not the exact long-range part, keeps every coarse tile exact; the
model's error only changes how smooth $X$ is. Both rank-$n_L$ updates are
formed in W's $(\mu,\nu)$ sharding with no collective.
