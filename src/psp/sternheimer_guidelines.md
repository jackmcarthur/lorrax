Below is a start-to-finish implementation guide for an agent.

---

# Guide: insulating Sternheimer implementation of the (G=0) source column (\chi_{G'0}(q,\omega)), with Schur-complement low-energy preconditioning and a (k)-derivative JVP through the converged solve

Assume:

* insulating system, (f_v=1,\ f_c=0),
* arbitrary coarse-grid (q),
* goal is the **single-source (G=0) column**
  [
  \chi_{G'0}(q,\omega),
  ]
  from which the head is (G'=0) and the wing is all (G'\neq 0),
* use cell-periodic Bloch states (u_{nk}),
* (H_k=e^{-ik\cdot r}He^{ik\cdot r}).

The efficient formulation is: for each ((v,k,q)), solve one Sternheimer equation in the shifted sector (p=k+q) with source (e^{iq\cdot r}u_{vk}), reconstruct the induced density, and Fourier-project it to all (G'). For (k)-derivatives of this (G=0) column, differentiate the **converged** Sternheimer equation implicitly instead of differentiating through iterations. The Schur-complement idea of Cancès et al. is to split off a small low-energy conduction subspace built from extra SCF orbitals and solve the remainder iteratively, improving stability and reducing Hamiltonian applications. ([arXiv][1])

## 1. Key equations

For source (G=0), the Sternheimer equation for each occupied state (u_{vk}) is
[
Q_p\big(H_p-\varepsilon_{vk}-\omega-i\eta\big)Q_p,
|\delta u_{vk}^{(q,0)}(\omega)\rangle
=====================================

-,Q_p,e^{iq\cdot r},|u_{vk}\rangle,
\qquad p=k+q,
]
with
[
Q_p=1-P^{\mathrm{occ}}*p,
\qquad
P^{\mathrm{occ}}*p=\sum*{v'} |u*{v'p}\rangle\langle u_{v'p}|.
]

Define the source vector
[
|b_{vk}^{(q)}\rangle \equiv Q_p e^{iq\cdot r}|u_{vk}\rangle.
]

The induced density for that source is
[
\delta n^{(q,0)}(r,\omega)
==========================

\sum_{vk}
\Big(
u_{vk}^*(r),\delta u_{vk}^{(q,0)}(r,\omega)
+
\delta u_{vk}^{(q,0)*}(r,-\omega),u_{vk}(r)
\Big).
]

The full (G'= ) output column is obtained by Fourier projection:
[
\boxed{
\chi_{G'0}(q,\omega)
====================

\int d r, e^{-i(q+G')\cdot r},\delta n^{(q,0)}(r,\omega).
}
]
So:

* the **head** is (G'=0),
* the **wing column** is all (G'\neq 0).

For (\omega=0), the resonant and antiresonant pieces combine, and the head may also be accumulated directly from the Sternheimer solution as
[
\chi_{00}(q,0)
==============

-2,\mathrm{Re}\sum_{vk}\langle b_{vk}^{(q)}|
(H_p-\varepsilon_{vk})^{-1}
|b_{vk}^{(q)}\rangle
]
inside the conduction space, up to your code’s sign convention for (\chi). The safer implementation is still to build (\delta n) and project it, since that gives all (G') at once.

## 2. Efficient source construction and output projection

In plane waves,
[
u_{vk}(r)=\sum_G c_{vk}(G)e^{iG\cdot r}.
]

To build the source (b_{vk}^{(q)}):

1. FFT (u_{vk}(G)\to u_{vk}(r)),
2. multiply by (e^{iq\cdot r}),
3. FFT back if needed,
4. project out the occupied space at (p=k+q):
   [
   Q_p|\phi\rangle = |\phi\rangle - U_p(U_p^\dagger \phi),
   ]
   where (U_p\in \mathbb C^{N_{\mathrm{PW}}\times N_v}) collects the occupied states at (p).

If solving many valence states at fixed ((k,q)), batch them:
[
B_k^{(q)} = Q_{k+q} e^{iqr} U_k^{(v)},
]
so the projection uses matrix-matrix multiplies.

Once the response orbitals are solved, build the induced density in real space by elementwise products (u^*,\delta u), sum over (v,k), and FFT/project to all (G'). One (G=0) source solve gives the entire (G'0) column, not just the head.

## 3. Schur-complement low-energy preconditioning

Use your standard Teter-like plane-wave preconditioner on the large residual space. On top of that, split off a small low-energy conduction subspace using extra SCF / Davidson orbitals. The 2023 idea is to use those extra orbitals to form a Schur complement, which improves conditioning and reduced Hamiltonian applications by about 40% in reported tests. ([arXiv][1])

At each shifted momentum (p), let (U_p^{\mathrm{extra}}\in\mathbb C^{N_{\mathrm{PW}}\times M}) be (M) extra conduction-like Ritz vectors. Define
[
T_p = U_p^{\mathrm{extra}}U_p^{\mathrm{extra}\dagger},
\qquad
R_p = 1 - P_p^{\mathrm{occ}} - T_p.
]

Write the Sternheimer unknown as
[
x = x_T + x_R,
\qquad x_T=T_px,\quad x_R=R_px,
]
and define
[
A_{vkp} \equiv H_p-\varepsilon_{vk}-\omega-i\eta.
]

In block form on (T_p\oplus R_p),
[
\begin{pmatrix}
A_{TT} & A_{TR}\
A_{RT} & A_{RR}
\end{pmatrix}
\begin{pmatrix}
x_T\
x_R
\end{pmatrix}
=============

\begin{pmatrix}
b_T\
b_R
\end{pmatrix}.
]

Eliminate the small (T)-block:
[
x_T = A_{TT}^{-1}(b_T-A_{TR}x_R),
]
[
\boxed{
\left(A_{RR}-A_{RT}A_{TT}^{-1}A_{TR}\right)x_R
==============================================

b_R-A_{RT}A_{TT}^{-1}b_T.
}
]

Then reconstruct (x_T). In practice:

* (A_{TT}) is tiny and dense, often close to diagonal in the extra-orbital basis,
* solve the Schur system for (x_R) with projected CG or MINRES at (\omega=0),
* keep the usual Teter/kinetic preconditioner on the (R)-part.

For a nearby (k) or for a derivative/JVP solve, use the explicit (T)-space solution as the initial guess:
[
x^{(0)} = U_p^{\mathrm{extra}},A_{TT}^{-1}b_T.
]
That is the immediately relevant “seed.”

## 4. Iterative solver details

For static insulating (\omega=0), inside the projected conduction space,
[
A_{vkp}=Q_p(H_p-\varepsilon_{vk})Q_p
]
is Hermitian positive definite, so use projected **CG** or **MINRES**.

Matrix-free application:
[
y = Q_p(H_p x - \varepsilon_{vk}x),
]
with (Q_p) applied every iteration or at least to residuals.

Preconditioner:

* use your existing Teter-like plane-wave preconditioner based on (|p+G|^2/2),
* apply it only on the (R)-space part in the Schur solve,
* do not bother trying to encode the exact (-\varepsilon_{vk}) shift in detail; the usual kinetic/Teter scaling is the right cheap default.

If (\omega+i\eta\neq 0), the operator becomes complex shifted and non-Hermitian; then switch to GMRES or BiCGSTAB, but for the requested (\omega=0) implementation CG/MINRES is the main path.

## 5. JVP through the converged (G=0) Sternheimer solve for (k)-derivatives

Do **not** differentiate through iterations. Use the converged linear solve as an implicit function.

Let
[
A(\theta)x(\theta)=b(\theta),
]
where for the (G=0) column,
[
A(\theta)=Q_{k+q}(H_{k+q}-\varepsilon_{vk}-\omega-i\eta)Q_{k+q},
\qquad
b(\theta)=-Q_{k+q}e^{iqr}u_{vk},
]
and (\theta) is the differentiated parameter, e.g. (k_i).

Then the JVP (\dot x = \partial_{\theta}x\cdot \dot\theta) is obtained from one more Sternheimer-like solve:
[
\boxed{
A,\dot x = \dot b - \dot A,x.
}
]

This is the key equation for the (k)-derivative of the (G=0) response column. Use the **same** Schur/Teter-preconditioned solver as in the primal solve, with right-hand side (\dot b-\dot A,x).

For (\theta=k_i), the required differentiated pieces are
[
\dot A =
(\partial_{k_i}Q_p)(H_p-\varepsilon_{vk}-\omega-i\eta)Q_p
+
Q_p(\partial_{k_i}H_p-\partial_{k_i}\varepsilon_{vk})Q_p
+
Q_p(H_p-\varepsilon_{vk}-\omega-i\eta)(\partial_{k_i}Q_p),
]
with (p=k+q), and
[
\dot b =
-(\partial_{k_i}Q_p)e^{iqr}u_{vk}
---------------------------------

Q_p e^{iqr}(\partial_{k_i}u_{vk}).
]
There is no (\partial_{k_i}e^{iqr}) term since (q) is held fixed here.

If you want the simplest first implementation, **freeze the projector** in the JVP:
[
\partial_{k_i}Q_p \approx 0,
]
and only include
[
\dot A \approx Q_p(\partial_{k_i}H_p-\partial_{k_i}\varepsilon_{vk})Q_p,
\qquad
\dot b \approx -Q_p e^{iqr}(\partial_{k_i}u_{vk}).
]
This already gives a useful first version if you have (\partial_{k_i}H_p), (\partial_{k_i}\varepsilon_{vk}), and (\partial_{k_i}u_{vk}) from autodiff / Sternheimer band-derivative machinery.

If you later want full consistency, include (\partial_{k_i}Q_p) by differentiating the occupied subspace at (p).

## 6. Finalizing the (k)-derivative of the (G=0) column

Once (\dot x = \partial_{k_i}\delta u_{vk}^{(q,0)}) is solved, differentiate the induced density:
[
\partial_{k_i}\delta n^{(q,0)}(r,\omega)
========================================

\sum_{vk}
\Big[
(\partial_{k_i}u_{vk})^*,\delta u_{vk}^{(q,0)}
+
u_{vk}^*,\partial_{k_i}\delta u_{vk}^{(q,0)}
+
\text{c.c.}
\Big].
]

Then project to all (G'):
[
\boxed{
\partial_{k_i}\chi_{G'0}(q,\omega)
==================================

\int d r, e^{-i(q+G')\cdot r},\partial_{k_i}\delta n^{(q,0)}(r,\omega).
}
]

So the JVP of the (G=0) source column is produced by:

1. primal Sternheimer solve for (\delta u),
2. one JVP Sternheimer solve for (\partial_{k_i}\delta u),
3. density contraction and FFT/projection.

This is the efficient “through the self-consistent solution” route.

## 7. Static (\omega=0) algorithm for arbitrary coarse-grid (q)

For each coarse-grid (q):

1. Initialize
   [
   \chi_{G'0}(q,0)=0,
   \qquad
   \partial_{k_i}\chi_{G'0}(q,0)=0
   ]
   if derivatives are needed.

2. For each (k):

   * set (p=k+q),
   * load occupied states (U_p),
   * load extra states (U_p^{\mathrm{extra}}) for the Schur block,
   * for each valence (v):

     * build
       [
       b_{vk}^{(q)} = Q_p e^{iqr}u_{vk},
       ]
     * solve the Schur-preconditioned Sternheimer equation
       [
       Q_p(H_p-\varepsilon_{vk})Q_p,x_{vk}^{(q)} = -b_{vk}^{(q)},
       ]
     * accumulate the primal induced-density contribution.

3. If (k)-derivatives are needed:

   * build
     [
     r^{(i)}*{vkq}=\dot b - \dot A,x*{vk}^{(q)},
     ]
   * solve
     [
     Q_p(H_p-\varepsilon_{vk})Q_p,\dot x_{vk}^{(q,i)} = r^{(i)}_{vkq},
     ]
     with the **same** Schur/Teter machinery,
   * accumulate the derivative induced-density contribution.

4. After summing over (v,k), FFT/project (\delta n) and (\partial_{k_i}\delta n) to all (G'):
   [
   \chi_{G'0}(q,0)
   ===============

   \int dr,e^{-i(q+G')r},\delta n^{(q,0)}(r,0),
   ]
   [
   \partial_{k_i}\chi_{G'0}(q,0)
   =============================

   \int dr,e^{-i(q+G')r},\partial_{k_i}\delta n^{(q,0)}(r,0).
   ]

Then:

* (G'=0) gives the **head**,
* (G'\neq 0) gives the **wing column**.

## 8. Pseudocode sketch

```python
for q in coarse_q_grid:
    chi_col_q = 0
    dki_chi_col_q = 0   # if needed

    for k in k_grid:
        p = k + q

        Uocc_p   = load_occ_states(p)
        Uextra_p = load_extra_states(p)   # Schur block
        Qp       = projector_from_occ(Uocc_p)

        for v in valence_bands:
            u_vk = load_state(k, v)
            eps_vk = load_eigenvalue(k, v)

            # build source b = Q_p exp(i q.r) u_vk
            b = project_Q(Qp, phase_shift_realspace(u_vk, q))

            # primal solve with Schur split + Teter preconditioner
            x = solve_sternheimer_schur(
                    H_p_apply,
                    eps_vk,
                    Qp,
                    Uextra_p,
                    rhs = -b,
                    omega = 0.0
                )

            accumulate_density_primal(u_vk, x)

            if need_k_deriv:
                du_dki_vk = load_or_compute_du_dki(k, v)
                deps_dki_vk = load_or_compute_deps_dki(k, v)

                # simplest first implementation: freeze dQ/dk = 0
                dA_x = project_Q(Qp,
                         apply_dH_dki_p(x) - deps_dki_vk * x
                       )
                db   = -project_Q(Qp, phase_shift_realspace(du_dki_vk, q))

                rhs_jvp = db - dA_x

                dx_dki = solve_sternheimer_schur(
                            H_p_apply,
                            eps_vk,
                            Qp,
                            Uextra_p,
                            rhs = rhs_jvp,
                            omega = 0.0
                         )

                accumulate_density_jvp(u_vk, du_dki_vk, x, dx_dki)

    chi_col_q     = fft_project_density_to_Gprime(delta_n_q)
    dki_chi_col_q = fft_project_density_to_Gprime(dki_delta_n_q)
```

## 9. Checks

I do have code in GWJAX that will calculate the explicit sum-over-states head correction using the S-tensor, but it is limited to a small included number of conduction bands in the sum over c instead of using all of them in sternheimer, so agreement will be flawed (you can do a few values and extrapolate; the fact that the head tensor has nine values may help you confirm if they track trendwise but are just missing prefactors, if they're off by a multiplicative constant ish or that kind of thing)
[
\chi_{G'0}(q,0)
===============

-2\sum_{vck}
\frac{
\langle vk|e^{-iqr}|c,k+q\rangle
\langle c,k+q|e^{i(q+G')r}|vk\rangle
}{
\varepsilon_{c,k+q}-\varepsilon_{vk}
}.
]

Check projector orthogonality:
[
U_p^\dagger b \approx 0,\qquad U_p^\dagger x \approx 0,\qquad U_p^\dagger \dot x \approx 0.
]

Check (q\to 0) behavior of the head:
[
\chi_{00}(q,0)\sim q_i S_{ij} q_j.
]

And verify that one (G=0) source solve reproduces the whole (G'0) column.

---

If you want, I can compress this one more notch into a short “agent-ready implementation checklist” with only equations and bullet steps.

[1]: https://arxiv.org/pdf/2210.04512?utm_source=chatgpt.com "arXiv:2210.04512v1 [math.NA] 10 Oct 2022"
