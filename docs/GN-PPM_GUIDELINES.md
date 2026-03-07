Below is a **drop-in “GN-PPM in the ISDF/collocation basis”** writeup: definitions + the exact algebra you’d code, plus **what people usually pick for the nonzero imaginary frequency**.

I’m going to be very explicit about **what is basis-covariant (matrix ops)** vs **what is not (the elementwise GN fit)**, because that matters for whether you reproduce the *plane-wave* GN-PPM exactly.

---

## 0) Notation and ISDF objects

Let (Z) denote your “interpolation-vector” map from the collocation basis to plane waves:
[
Z_{G,\mu} \equiv \zeta_{q,\mu}(G),\qquad
Z^\dagger_{\mu,G} \equiv \zeta^*_{q,\mu}(G).
]

Your exact identities (given your chosen ({\zeta_{q,\mu}}) and collocation points) are:

* **Irreducible polarizability (RPA)**
  [
  \chi^0_{GG'}(\omega)=\sum_{\mu\nu} Z_{G,\mu},\chi^0_{\mu\nu}(\omega),Z^\dagger_{\nu,G'}.
  ]

* **Coulomb in the ISDF basis**
  [
  v_{\mu\nu}(q)=\sum_{GG'} Z^\dagger_{\mu,G},\frac{4\pi}{|q+G|^2},\delta_{GG'},Z_{G',\nu}
  = (Z^\dagger v Z)_{\mu\nu}.
  ]

Here (\chi^0_{\mu\nu}(\omega)) is computed from the **collocated** pair densities (M_{cvkq}(r_\mu)) exactly as you wrote (just replace the plane-wave (M(G)) by (M(r_\mu))).

---

## 1) Exact screened interaction in the ISDF basis (no PPM yet)

Define the dielectric matrix in the ISDF basis
[
\varepsilon_{\mu\nu}(\omega) \equiv \delta_{\mu\nu} - \sum_\eta v_{\mu\eta},\chi^0_{\eta\nu}(\omega)
\quad\Longleftrightarrow\quad
\varepsilon(\omega)=I - v,\chi^0(\omega).
]

Then the **exact** screened Coulomb in that same basis is
[
W_{\mu\nu}(\omega)=\sum_\eta \varepsilon^{-1}*{\mu\eta}(\omega),v*{\eta\nu}.
]

Equivalently, with your (\Pi) definition,
[
\Pi(\omega) \equiv \chi^0(\omega),[I - v\chi^0(\omega)]^{-1}
]
and
[
W(\omega)=v + v,\Pi(\omega),v.
]

Everything above is **basis-covariant**: you can do it in ({G}), in ({\mu}), in any AO/RI basis—same algebra.

---

## 2) GN-PPM “two-point” fit written in the ISDF basis

The standard Godby–Needs PPM is usually stated as an **elementwise** (Hadamard) fit of a matrix-valued response at two frequencies (z_1=0) and (z_2=i\omega_p).

You can literally write the same thing in the ISDF basis by replacing (GG'\to \mu\nu).

### Step 2.1: compute the two “target” matrices

Pick (\omega_p>0) and compute
[
\chi^0(0),\qquad \chi^0(i\omega_p),
]
then build
[
\Pi(0)=\chi^0(0),[I-v\chi^0(0)]^{-1},\qquad
\Pi(i\omega_p)=\chi^0(i\omega_p),[I-v\chi^0(i\omega_p)]^{-1}.
]

### Step 2.2: define GN pole parameters **elementwise** in (\mu\nu)

The GN ansatz (in the form you quoted) is
[
\Pi_{\mu\nu}(\omega)\approx \frac{2,B_{\mu\nu},\Omega_{\mu\nu}}{\omega^2-\Omega_{\mu\nu}^2}.
]

Then the two-point GN parameter extraction (same structure as your (GG') equations) is

[
\Omega_{\mu\nu}
===============

\omega_p,
\sqrt{
\operatorname{Re}
\left[
\frac{\Pi_{\mu\nu}(i\omega_p)}
{\Pi_{\mu\nu}(0)-\Pi_{\mu\nu}(i\omega_p)}
\right]
},
]
[
B_{\mu\nu}=-\frac{1}{2},\Pi_{\mu\nu}(0),\Omega_{\mu\nu}.
]

(Implementation detail: codes usually guard against noisy/small denominators and enforce (\Omega_{\mu\nu}>0) by clipping the argument of the square root.)

### Step 2.3: evaluate (\Pi^{\text{GN}}(\omega)) and (W^{\text{GN}}(\omega)) in (\mu)-space

For any (\omega),
[
\Pi^{\text{GN}}*{\mu\nu}(\omega)=\frac{2,B*{\mu\nu},\Omega_{\mu\nu}}{\omega^2-\Omega_{\mu\nu}^2}
]
and
[
W^{\text{GN}}(\omega)=v + v,\Pi^{\text{GN}}(\omega),v.
]

That’s the “everything needed to implement” version in the ISDF basis.

---
## 3) What (\omega_p) do people use?

There are (at least) two common “in practice” choices:

1. **Fixed default “plasmon-pole energy”** used by some workflows/tutorials.
   For example, Yambo tutorials commonly show a default pole energy around **27.211 eV (≈ 1 Hartree)** (their `PPAPntXp`) and note it’s not usually changed. ([ENCCS][1])

2. **Choose (\omega_p) near a characteristic plasmon frequency**, often quoted as order **0.5 Hartree** in teaching material.
   ABINIT’s GW tutorial notes that the nonzero frequency in the Godby model is recommended close to the plasmon frequency, and says plasmons are “usually close to 0.5 Hartree.” ([Abinit][2])

So, a reasonable starting bracket many people try is:

* (\omega_p \sim 0.5\ \text{Ha}) (≈ 13.6 eV) up to
* (\omega_p \sim 1.0\ \text{Ha}) (≈ 27.2 eV),

with code-specific defaults (e.g. Yambo’s example default).

---
