# The S-tensor convention

Several places in LORRAX build a symmetric 3×3 tensor that describes how a
scalar q-space quantity behaves as **q** approaches the zone centre. They are
not the same object unless they agree on two things: the frame the indices live
in, and the power of q the tensor is the coefficient of. This page fixes both,
once, for the whole tree.

## The canonical form

The density-response S-tensor is the **Cartesian q²-coefficient**:

$$
\chi_{G'=0}(\mathbf{q}\to 0,\ \omega) \;=\; q_a\, S_{ab}(\omega)\, q_b ,
$$

with **q** in Cartesian reciprocal units (1/bohr, i.e. `blat * bvec` applied to
the crystal coordinates) and `S` a symmetric complex `(3, 3)` array per
frequency.

`common.chi_from_dipole.compute_S_omega` returns exactly this, and it is the
builder every reader in the tree is written against:

- `gw.head_correction.from_s_tensor` passes the result through unchanged as
  `S_cart`;
- `vcoul.Bulk3D.q0_average` contracts it as
  `einsum('qi,ij,qj->q', rq, S, rq)` against Cartesian mini-BZ draws `rq` and
  forms the screened head `⟨v / (1 − v·qSq)⟩`, which is `W = v/ε` with
  `ε = 1 − v χ`.

So the reader's own arithmetic pins the convention: `qSq` has to *be* `χ₀₀`,
not twice it, and `q` has to be Cartesian because `rq` is.

## The second builder, and what it used to ship

`psp.run_sternheimer.compute_s_tensor_contrib_at_q0` computes the same physical
object by a completely independent route — three Sternheimer solves per k, no
sum over states, no dipole file. That is valuable precisely because it is
independent: the two routes disagreeing is a measurement of an approximation,
not a bug hunt.

But its kernel differentiates with respect to the *crystal* `kvec` the traced
operator is built at, so what falls out is the **crystal-coordinate Hessian**

$$
H_{ij} \;=\; \frac{\partial^2 \chi_{00}}{\partial q_i \partial q_j}\bigg|_0 ,
\qquad
\chi_{00} = \tfrac12\, q^{\rm crys}_i H_{ij} q^{\rm crys}_j .
$$

That differs from the canonical form by a factor of two *and* by a frame
change. Before 2026-08-09 the raw Hessian was written to
`sternheimer.h5:s_tensor_q0` under a note that said so, and nothing in the tree
read it — so the two builders existed side by side with no way to consume both
consistently, and any third tensor added to the family would have inherited the
ambiguity. That is `SMALL_ISSUES.md` row 22.

**The ruling: the Cartesian q²-coefficient is canonical, because it is the one
with readers.** `run_sternheimer` now converts before writing, at the one site
that has the lattice in hand:

$$
S_{ab} \;=\; \tfrac12 \big(B^{-1} H\, B^{-\mathsf{T}}\big)_{ab},
\qquad B = \texttt{blat}\cdot\texttt{bvec} \ \ (\text{rows Cartesian}),
$$

the ½ undoing the Hessian and `B⁻¹` carrying `q_crys = q_cart · B⁻¹`. The
dataset on disk is canonical, it carries an explicit
`s_tensor_convention = "cartesian_q2_coefficient"` attribute, and the raw
Hessian no longer leaves the driver. The two routes are now directly
comparable, which is the cross-check the Sternheimer path was worth keeping
for.

## The rule for anything new

A new rank-two q-space tensor in this family must be Cartesian, and must say in
its docstring which power of q it is the coefficient of. The next one is the
mini-BZ exchange-head moment `M_ab = ⟨v(q) q_a q_b⟩_cell`, which is the
coefficient of the *dipole* bilinear rather than of q itself — a different
power, correctly declared, and therefore not a hazard.
