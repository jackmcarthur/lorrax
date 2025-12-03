Yeah, your memory is basically right: the Hybertsen–Louie generalized plasmon-pole (HL-GPP) model is literally constructed so that a **single pole** per ((q,G,G')) reproduces specific **frequency moments** (ω and ω⁻¹) of the *exact* response, plus the static dielectric matrix.

Let me split this into:

1. Where an actual derivation is written down
2. The math of the derivation in compact form
3. Exactly which conditions / moments are enforced

---

## 1. Where people actually derive HL-GPP

Hybertsen & Louie state the idea in their band-gap paper and the model-dielectric paper, but they don’t walk you through the full algebra. The details are fleshed out in later work and reviews:

* **Original HL papers**

  * Hybertsen & Louie, *Electron correlation in semiconductors and insulators* (PRB 34, 5390 (1986)): introduces the GPP model and says explicitly that it is “based on exact sum rules” and “reproduces the ω and ω⁻¹ moments of the exact many-body response function.” ([Bohrium][1])
  * Hybertsen & Louie, *Model dielectric matrices for quasiparticle self-energy calculations* (PRB 37, 2733 (1988)): more focused on the model ε⁻¹ itself. ([Nature][2])

* **Explicit derivation / explanation of the constraints**

  * Soininen et al., *Electron self-energy calculation using a general multi-pole approximation* (J. Phys.: Condens. Matter 15, 2573 (2003)) – section on plasmon-pole models spells out that typical models (incl. HL) are obtained by enforcing **static screening** and the **f-sum rule** on ε or ε⁻¹. ([NIST][3])
  * Bruneval, *Accurate self-energies in a plane-wave basis…* (PRB 78, 085125 (2008)) – explicitly notes that a sum rule relation is what allows Hybertsen and Louie to fix the free parameters of their model for ε⁻¹. ([APS Link][4])
  * Rangel et al., *Reproducibility in G₀W₀ calculations for solids* (2020) – has a clear text summary: in the HL approach “the PPM parameters are determined from sum rules and the static dielectric matrix.” ([Dial][5])
  * Leon et al., *Frequency dependence in GW made simple using a multipole approximation* (2021) – their Appendix (explicitly mentioned) lists the **sum-rule constraints imposed by the Hybertsen–Louie scheme** and then generalizes them to multiple poles. ([Iris][6])

Pretty much every modern GW review that mentions HL-GPP repeats the same line:

> *The model reproduces the ω and ω⁻¹ moments of the exact many-body response function.* ([American Chemical Society Publications][7])

So yes: what you heard about moments is not folklore, it’s literally how they define the model.

---

## 2. Compact derivation for a single matrix element

Take one matrix element of the **inverse dielectric** (the thing that actually appears in $W = \epsilon^{-1} v$):

$$
\epsilon^{-1}_{GG'}(q,\omega).
$$

HL assume for each ((q,G,G')) that the **frequency dependence** can be represented by a single symmetric pole pair:

$$
\epsilon^{-1}*{GG'}(q,\omega)
:= \delta*{GG'} +
\frac{\Omega^2_{GG'}(q)}{\omega^2 - \tilde{\omega}^2_{GG'}(q) + i,\eta},
$$

with two unknown real parameters per element:

* $\Omega^2_{GG'}(q)$ = “oscillator strength”
* $\tilde{\omega}_{GG'}(q)$ = effective plasmon frequency

This functional form automatically ensures:

* correct **analytic structure** (causality / KK),
* symmetry $\epsilon^{-1}(\omega) = [\epsilon^{-1}(-\omega)]^*$,
* and the high-frequency limit $\epsilon^{-1}(\omega \to \infty) \to \delta_{GG'}$.

The **imaginary part** (retarded or time-ordered differ only by (i\eta) convention) is then a pair of δ-peaks:

$$
\operatorname{Im}\epsilon^{-1}*{GG'}(q,\omega)
:= -\frac{\pi \Omega^2*{GG'}(q)}{2\tilde{\omega}*{GG'}(q)}
\left[\delta(\omega-\tilde{\omega}*{GG'}) - \delta(\omega+\tilde{\omega}_{GG'})\right].
$$

Define the spectral function
$$
S_{GG'}(q,\omega) = -\frac{1}{\pi}\operatorname{Im}\epsilon^{-1}_{GG'}(q,\omega).
$$

Within the single-pole ansatz (restricting to $\omega>0$):

* **ω-moment**:
  $$
  M_1^{\text{(model)}}(q,GG')
  = \frac{2}{\pi} \int_0^\infty d\omega,\omega,\big(-\operatorname{Im}\epsilon^{-1}\big)
  = \Omega^2_{GG'}(q).
  $$

* **ω⁻¹-moment**:
  $$
  M_{-1}^{\text{(model)}}(q,GG')
  = \frac{2}{\pi} \int_0^\infty d\omega,\frac{1}{\omega},\big(-\operatorname{Im}\epsilon^{-1}\big)
  = \frac{\Omega^2_{GG'}(q)}{\tilde{\omega}^2_{GG'}(q)}.
  $$

And from KK, the **static limit** is

$$
\epsilon^{-1}*{GG'}(q,0)
:= \delta*{GG'} - \frac{\Omega^2_{GG'}(q)}{\tilde{\omega}^2_{GG'}(q)}
:= \delta_{GG'} - M_{-1}^{\text{(model)}}(q,GG').
$$

So with the single pole, specifying either

* $\epsilon^{-1}(q,0)$ and $M_1$, or
* $M_1$ and $M_{-1}$

is equivalent to specifying $\Omega^2$ and $\tilde{\omega}$.

---

## 3. What HL actually match to the *exact* response

Now, what do they match these model moments to?

Hybertsen & Louie compute:

1. **Static inverse dielectric matrix**
   $\epsilon^{-1}_{GG'}(q,0)$ in RPA, from a DFT density-response calculation. This is the “static dielectric matrices” paper (PRB 35, 5585 (1987)). ([gpaw.readthedocs.io][8])

2. **Frequency sum rule (f-sum rule)**
   They use an exact relation for the **ω-moment** (the f-sum rule) of the density–density response / dielectric function for each ((q,G,G')). This is what Soininen calls “modifying [the electron gas form] so that the f-sum rule and static dielectric screening are correctly given by the model.” ([NIST][3])

Putting that in the same notation as above:

* Let $M_1^{\text{(exact)}}(q,GG')$ be the ω-moment of the *exact* $-\operatorname{Im}\epsilon^{-1}$ (or equivalently of $-\operatorname{Im}W$, since v is frequency-independent). This is tied to the f-sum rule / charge conservation, and they evaluate it using the ground-state electron density and pseudopotential charge, not by explicit frequency integration. ([APS Link][4])

* Let $\epsilon^{-1,\text{exact}}_{GG'}(q,0)$ be the RPA static matrix element.

Then the HL constraints are:

$$
\boxed{
\begin{aligned}
M_1^{\text{(model)}}(q,GG') &= M_1^{\text{(exact)}}(q,GG'), \
\epsilon^{-1,\text{model}}*{GG'}(q,0) &= \epsilon^{-1,\text{exact}}*{GG'}(q,0).
\end{aligned}
}
$$

Using the relations above, this means:

$$
\boxed{
\begin{aligned}
\Omega^2_{GG'}(q) &= M_1^{\text{(exact)}}(q,GG'), \
\frac{\Omega^2_{GG'}(q)}{\tilde{\omega}^2_{GG'}(q)} &=
\delta_{GG'} - \epsilon^{-1,\text{exact}}_{GG'}(q,0).
\end{aligned}
}
$$

So, in words:

> **For each ((q,G,G')), HL choose a single pole whose strength and position are fixed such that:**
>
> * the model reproduces the exact **ω-moment** (f-sum rule) of the loss function, and
> * the model reproduces the exact **ω⁻¹-moment**, which is equivalent (via KK) to reproducing the *static* inverse dielectric matrix.

That’s exactly what is meant when papers say “the model reproduces the ω and ω⁻¹ moments of the exact many-body response function.” ([Bohrium][1])

From there, you just plug this ε⁻¹ into

$$
W_{GG'}(q,\omega) = \epsilon^{-1}*{GG'}(q,\omega),v*{G'}(q)
$$

and perform the $\omega$ integral in Σ analytically; that’s the usual “GW with plasmon-pole” trick.

---

### TL;DR in your language

* Yes, HL-GPP is explicitly a **moment-matching** PPM.
* It’s built so that for each ((q,G,G')) the **single pole** reproduces:

  * the **static** ε⁻¹ matrix element, and
  * the **ω and ω⁻¹ moments** (i.e. the f-sum rule and its KK partner) of the *exact* response.
* Those two conditions are enough to fix the pole strength and position; everything else (KK, causality, high-ω limit) is baked into the ansatz.

If you want, next step we can write the corresponding **W(r,r',ω)** with the HL pole parameters explicitly and show how those same moments look in real space.

[1]: https://www.bohrium.com/paper-details/electron-correlation-in-semiconductors-and-insulators-band-gaps-and-quasiparticle-energies/814644991686082562-529?utm_source=chatgpt.com "Electron correlation in semiconductors and insulators"
[2]: https://www.nature.com/articles/s41524-023-01136-y?utm_source=chatgpt.com "Accelerating GW calculations through machine-learned ..."
[3]: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=841698&utm_source=chatgpt.com "Electron self-energy calculation using a general multi-pole ..."
[4]: https://link.aps.org/doi/10.1103/PhysRevB.78.085125?utm_source=chatgpt.com "Accurate self-energies in a plane-wave basis using only a few ..."
[5]: https://dial.uclouvain.be/pr/boreal/object/boreal%3A231673/datastream/PDF_01/view?utm_source=chatgpt.com "Reproducibility in G0W0 calculations for solids"
[6]: https://iris.cnr.it/retrieve/d381cc8b-824e-4e5a-91b9-caa84940915f/Frequency%20dependence%20in%20GW%20made%20simple%20using%20a%20multi-pole%20approximation.pdf?utm_source=chatgpt.com "arXiv:2109.01532v1 [cond-mat.mtrl-sci] 3 Sep 2021"
[7]: https://pubs.acs.org/doi/10.1021/ct500958p?utm_source=chatgpt.com "Large Scale GW Calculations | Journal of Chemical Theory ..."
[8]: https://gpaw.readthedocs.io/documentation/tddft/dielectric_response.html?utm_source=chatgpt.com "Linear dielectric response of an extended system: theory"
