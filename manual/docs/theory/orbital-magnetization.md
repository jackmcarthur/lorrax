# Orbital magnetization

`psp.orbital_response.orbital_magnetization` The modern-theory orbital moment of each band and of the unit cell, from the stored velocity matrix of a spinor wavefunction.
{: .page-summary }

<div class="section-cards"></div>

!!! warning "Under development"
    The orbital magnetization feature is under development. Its interface and output may change between releases, and its results should be checked independently before they are used in a publication.

## <span class="n">1</span> The modern theory {: #om-theory }

The orbital moment of a crystal cannot be computed as the expectation value of $\hat{\mathbf r}\times\hat{\mathbf p}$, because the position operator is not defined between Bloch states. The modern theory of orbital magnetization writes it instead through the $\bk$ derivatives of the cell-periodic states, and LORRAX evaluates it in a sum-over-states form built from the velocity matrix.
{: .intro }

### <span class="n">1.1</span> The bulk formula {: #om-bulk }

For electrons of charge $-e$ the orbital magnetization is a Brillouin-zone integral over the occupied states [1–5]. With $H_\bk=e^{-i\bk\cdot\br}He^{i\bk\cdot\br}$ and $\mu$ the chemical potential,

$$
\mathbf M=\frac{e}{2\hbar}\,\mathrm{Im}\sum_n\int\!\frac{d^dk}{(2\pi)^d}\,f_{n\bk}\,\langle\nabla_\bk u_{n\bk}|\times\big(H_\bk+E_{n\bk}-2\mu\big)|\nabla_\bk u_{n\bk}\rangle \tag{1.1}
$$

Splitting $H_\bk+E_{n\bk}-2\mu$ into $(H_\bk-E_{n\bk})+2(E_{n\bk}-\mu)$ separates the self-rotation of a wavepacket from the motion of its centre, which is weighted by the Berry curvature:

$$
\mathbf m_{n\bk}=\frac{e}{2\hbar}\,\mathrm{Im}\,\langle\nabla_\bk u_{n\bk}|\times\big(H_\bk-E_{n\bk}\big)|\nabla_\bk u_{n\bk}\rangle,\qquad
\boldsymbol\Omega_{n\bk}=-\mathrm{Im}\,\langle\nabla_\bk u_{n\bk}|\times|\nabla_\bk u_{n\bk}\rangle \tag{1.2}
$$

so that $\mathbf M=\sum_n\int\frac{d^dk}{(2\pi)^d}f_{n\bk}\big[\mathbf m_{n\bk}+\frac{e}{\hbar}\boldsymbol\Omega_{n\bk}(\mu-E_{n\bk})\big]$ at zero temperature. We call $\mathbf m_{n\bk}$ the orbital moment of band $n$. It is the quantity that colours the band structure, while only the combination (1.1) is the thermodynamic magnetization. Without spin–orbit coupling a collinear magnet splits into two spinless channels in which $\boldsymbol\Omega_{n\bk}$ and $\mathbf m_{n\bk}$ are odd in $\bk$, and $\mathbf M$ vanishes. The feature therefore needs a two-component spinor WFN.

### <span class="n">1.2</span> Sum over states {: #om-sos }

`psp.orbital_response.orbital_moments`
{: .in-code }

First-order perturbation theory expresses the interband part of $\nabla_\bk u_{n\bk}$ through the velocity $\mathbf v_{mn}=\langle m\bk|\hat{\mathbf v}|n\bk\rangle$ of the [velocity operator page](velocity-operator.md), with $\hat{\mathbf v}=\partial H_\bk/\partial\bk$:

$$
\langle u_{m\bk}|\nabla_\bk u_{n\bk}\rangle=\frac{\mathbf v_{mn}}{E_{n\bk}-E_{m\bk}},\qquad
A_{mn}=i\langle u_{m\bk}|\nabla_\bk u_{n\bk}\rangle=\frac{i\,\mathbf v_{mn}}{E_{n\bk}-E_{m\bk}},\qquad m\neq n \tag{1.3}
$$

The interband Berry connection is therefore read from the velocity, and no state is differenced on the k grid. Inserting a complete set of states into (1.2) and using (1.3) gives, in Rydberg atomic units with $\mathbf v$ in Ry bohr and energies in Ry,

$$
\frac{\mathbf m_{n\bk}}{\mu_B}=-\frac12\,\mathrm{Im}\sum_{m\neq n}\frac{\mathbf v_{nm}\times\mathbf v_{mn}}{E_{n\bk}-E_{m\bk}},\qquad
\boldsymbol\Omega_{n\bk}=-\mathrm{Im}\sum_{m\neq n}\frac{\mathbf v_{nm}\times\mathbf v_{mn}}{(E_{n\bk}-E_{m\bk})^2} \tag{1.4}
$$

with $\boldsymbol\Omega_{n\bk}$ in bohr². The factor $\tfrac12$ is $e/2\hbar$ divided by $\mu_B$ in these units. With it the atomic limit $\mathbf v_{nm}=i(E_n-E_m)\mathbf r_{nm}$ recovers the electron's $\mathbf m=-\mu_B\mathbf L/\hbar$. Since (1.4) contains no spin operator, the orientation of the orbital moment relative to the spin moment is a result of the calculation.

### <span class="n">1.3</span> The moment per cell {: #om-cell }

`bandstructure.orbital.orbital_totals`
{: .in-code }

On a uniform grid the integral becomes a sum with weights $w_\bk$ that add to one. The moment per cell is $\mathbf M$ times the cell volume, so the volume cancels. At a Fermi–Dirac temperature $T$ the code evaluates

$$
\frac{\mathbf m^{\mathrm{orb}}}{\mu_B}=\sum_\bk w_\bk\sum_n\Big[f_{n\bk}\,\frac{\mathbf m_{n\bk}}{\mu_B}+\boldsymbol\Omega_{n\bk}\,g(E_{n\bk})\Big],\qquad
g(E)=k_BT\ln\!\big(1+e^{(\mu-E)/k_BT}\big) \tag{1.5}
$$

where $\mu$ is the fixed-electron-count Fermi level. As $T\to0$, $g(E)$ tends to $(\mu-E)\,\theta(\mu-E)$, and (1.4) and (1.5) combine into the single sum

$$
\frac{\mathbf m^{\mathrm{orb}}}{\mu_B}=\frac12\sum_\bk w_\bk\,\mathrm{Im}\sum_{n\,\in\,\mathrm{occ}}\;\sum_{m\neq n}\frac{(\mathbf v_{nm}\times\mathbf v_{mn})\,(E_{n\bk}+E_{m\bk}-2\mu)}{(E_{n\bk}-E_{m\bk})^2} \tag{1.6}
$$

with $\mu$ at midgap. In an insulator whose occupied bands carry a nonzero Chern number the result depends on where $\mu$ sits in the gap, so the [standalone command](#om-cli) can report it with `--mu-scan` at the valence band maximum, midgap and the conduction band minimum.

### <span class="n">1.4</span> Which bands enter {: #om-bands }

The outer sum in (1.6) runs over occupied bands and the inner sum over every band the velocity file holds. That count is the stored band ceiling. Because each term falls off as $1/|E_{n\bk}-E_{m\bk}|$ for distant bands, the truncated sum converges as the ceiling rises, and the report prints the ceiling together with the distance $E_c-\mu$ of the highest stored band from $\mu$. Transitions with $|E_{n\bk}-E_{m\bk}|\le\delta$, where $\delta=1.4$ meV, are left out. Inside a degenerate multiplet the moment of a single band depends on the gauge, but the two omitted terms of each pair cancel as the splitting goes to zero, so the per-cell total of a complete multiplet is unaffected. On the band path the colour $\hat{\mathbf n}\cdot\mathbf m_{n\bk}$ is averaged over each degenerate set.

## <span class="n">2</span> Operators in the interpolation basis {: #om-basis }

The band operators are known on the coarse grid only. They reach an arbitrary $\bk$ through the same basis that the H-transform uses for the Hamiltonian.
{: .intro }

### <span class="n">2.1</span> A k-independent basis {: #om-galerkin }

The H-transform interpolation of Wu, Li, Wu, Lin, Hu and Yang [6] expands every state of the fitted window $S$ at every coarse $\bk_{\mathrm{co}}$ in one $\bk$-independent basis $\{Q_\alpha\}$. The basis is selected from the coarse-grid states themselves by randomized QR factorization with column pivoting and then orthonormalized. LORRAX calls it the Galerkin basis, fits it in `isdf/galerkin.py` and keeps it in `galerkin_dft.h5`. The [htransform utility](../utilities/htransform.md#how) describes the fit and the interpolation of $f(H)$.

### <span class="n">2.2</span> Band operators in the basis {: #om-operators }

`bandstructure.orbital.interpolate_band_operator`
{: .in-code }

An operator $\hat O$ whose band matrix is known on the coarse full zone is projected on the window and written in the basis as

$$
O_{\bk_{\mathrm{co}}}(\alpha,\beta)=\sum_{n,m\in S}\langle Q_\alpha|n\bk_{\mathrm{co}}\rangle\,\langle n\bk_{\mathrm{co}}|\hat O|m\bk_{\mathrm{co}}\rangle\,\langle m\bk_{\mathrm{co}}|Q_\beta\rangle \tag{2.1}
$$

The operators in use are $\sigma_a$, a projector on the Löwdin-orthogonalized atomic orbitals of one angular-momentum channel, and the three Cartesian components of the velocity. Each is Fourier transformed to lattice vectors and summed with phases at any $\bk_{\mathrm{fi}}$ in the same way as $f(H)$, and is then rotated into the eigenvectors of the interpolated $f(H_{\bk_{\mathrm{fi}}})$. On the band path the interpolated velocity and energies enter (1.4) to give $\mathbf m_{n\bk_{\mathrm{fi}}}$. On a uniform grid the same route carries $\sigma_a$ alone. Under `--qp-rotations` the rotation is unitary on the window, so the operators keep the unrotated WFN coefficients and only $f(H)$ takes the QP rotation. The per-cell total of §1.3 does not use the basis. It is summed on the coarse grid from the stored velocity at the irreducible points, with star weights, and averaged over the axial, time-odd action of the point group.

## <span class="n">3</span> Running {: #om-run }

The orbital moment is an option of the band-structure utility, and a standalone command evaluates (1.6) for an insulator.
{: .intro }

### <span class="n">3.1</span> With htransform {: #om-htransform }

`bandstructure.htransform.band_character_and_moments`
{: .in-code }

The velocity comes from the [dipole utility](../utilities/dipole.md), run on the deck's own WFN. The band-structure run then names it:

```bash
python -m psp.get_dipole_mtxels -i ht.in                       # writes dipole.h5
python -m bandstructure.htransform -i ht.in --velocity dipole.h5 --color orbital --color spin
```

| Flag or key | Effect |
|---|---|
| `--velocity dipole.h5` | The velocity of the WFN's own states. Enables the per-cell totals. Its basis and WFN fingerprint are checked. |
| `--color orbital` | Colours the path by $\hat{\mathbf n}\cdot\mathbf m_{n\bk}$, clipped at its 95th percentile. Needs `--velocity`. The other colours are on the [band-structure page](../utilities/htransform-bands.md#character). |
| `--moments-grid NX NY NZ` | Interpolates $H$ and $\sigma_a$ to a uniform grid and sums $\sum_{\mathrm{occ}}\langle\boldsymbol\sigma\rangle$ there, with $E_F$ found again on the grid. |
| `occ_smearing_width_ry` | Fermi–Dirac $k_BT$ of the moment sums. Unset, the orbital total is taken at $T=0$ with $\mu$ at midgap and the spin sums use a $10^{-4}$ Ry step. |

The magnetization axis $\hat{\mathbf n}$ is read from the Quantum ESPRESSO schema that authenticates the WFN, from the output total magnetization when it is nonzero and otherwise from the input starting-magnetization direction. Without a schema the axis is $z$. Alongside the orbital total the report prints the spin moment $-(g_e/2)\,\mu_B\sum_{\mathrm{occ}}\langle\boldsymbol\sigma\rangle$ with $g_e=2.00232$, and the sum of the two along $\hat{\mathbf n}$, all in $\mu_B$ per cell.

For QSGW the run reads the self-consistent run's `WFN_qp.h5` with its `dipole_qsgw.h5`, whose velocity $U^\dagger(\mathbf v^{\mathrm{DFT}}+D_\bk\Delta H)U$ includes the self-energy term ([Velocity operator §2.3](velocity-operator.md#forms-qp)). The command is `python -m bandstructure.htransform -i ht.in -wfn WFN_qp.h5 --velocity dipole_qsgw.h5`. `--velocity` refuses next to `--qp-rotations` or `--eqp-file`, because the DFT velocity divided by QP energy differences is not $\partial H^{\mathrm{QP}}/\partial\bk$.

| Output | Content |
|---|---|
| `bands_orbital.png` | path bands coloured by $\hat{\mathbf n}\cdot\mathbf m_{n\bk}$ in $\mu_B$, energies relative to the coarse-grid $E_F$ |
| `band_operators_path.npz` | path operators, $\mathbf m_{n\bk}$ on the path (`orbital_moment_path`), the colour values, $\hat{\mathbf n}$ and $\delta$ |
| `moments.txt` | grid spin sums and the Fermi level (with `--moments-grid`) |
| `htransform.out` | "Band character and moments": coarse spin sums, orbital total at the stored band ceiling, spin moment and their sum |

### <span class="n">3.2</span> The standalone command {: #om-cli }

`psp.orbital_magnetization.main`
{: .in-code }

`python -m psp.orbital_magnetization --wfn WFN.h5` evaluates (1.6) on the full zone of a DFT spinor WFN, at $T=0$ with an integer number of occupied bands, and needs no deck. The velocity comes from `velocity_only.h5` in the WFN's directory when that file exists. The dipole utility writes it with

```bash
python -m psp.get_dipole_mtxels -i gw.in --parallel-transport-out velocity_only.h5 --parallel-transport-velocity-only
```

and its WFN fingerprint, k grid and band range are checked on read. Without the file the command runs the velocity sweep itself from the `*.upf` files.

The command prints the band edges, the indirect gap and $\mu$, then a Hellmann–Feynman check of the loaded velocity (the RMS of $\mathrm{Re}\,v_{nn}-\partial E_{n\bk}/\partial\bk$ against the RMS slope, a diagnostic only), the spin moment $-\mu_B\sum_{\mathrm{occ}}\langle\boldsymbol\sigma\rangle$ with $g=2$, and the orbital moment per cell. The orbital moment is given in Cartesian components and along the spin-moment axis, labelled parallel or antiparallel to the spin. On a nonmagnetic WFN that axis is zero, and so is the projected value. Everything goes to standard output unless `--out` is given.

| Flag | Default | Meaning |
|---|---|---|
| `--wfn` | required | DFT `WFN.h5` with `nspinor = 2`. A QP WFN refuses. |
| `--nbnd` | the WFN's band count | Inner-sum band ceiling, clipped to the WFN's bands. With `velocity_only.h5` it may not exceed the bands stored there, so a file with fewer bands than the WFN needs an explicit `--nbnd`. |
| `--nocc` | the WFN's `nelec` | Number of occupied bands, one per electron on a spinor WFN. |
| `--mu` | midgap | $\mu$ in eV. Midgap is halfway between the valence maximum and conduction minimum over the grid. |
| `--mu-scan` | off | Also prints the moment along the spin axis at $\mu$ = valence maximum, midgap and conduction minimum. |
| `--deps-tol` | `1.4e-3` | Degeneracy tolerance $\delta$ in eV. |
| `--pseudo-dir` | WFN directory, then `../qe/scf`, `../qe/nscf` | Searched first when given. The `*.upf` files are needed only on the sweep route. |
| `--convergence` | off | Prints the moment along the spin axis with the inner sum cut at 50, 70, 85 and 100 % of `--nbnd`. |
| `--per-band` | off | Prints the twelve largest occupied-band contributions along the spin axis. |
| `--out` | none | `.npz` with the energies, $\mu$, `nocc`, the orbital and spin moments, the spin axis, the band-resolved partial sums (`colA_n`, `colB_n`, `cA`, `cB`) and the velocity source. |

## <span class="n">4</span> Refusals {: #om-refusals }

!!! refuses
    - A WFN with one spinor component.
    - `--color orbital` without `--velocity`.
    - `--velocity` with `--qp-rotations` or `--eqp-file`. Run on the SC run's `WFN_qp.h5` with `dipole_qsgw.h5` instead.
    - A velocity file of another basis or another WFN (`GATE dipole_basis`, `GATE velocity_wfn_fingerprint`), or a QP-basis velocity without its self-energy term (`GATE qp_velocity_sigma_term`).
    - A $T=0$ total without an integer electron count below a gap (`GATE orbital_totals_t0_gap`). Set `occ_smearing_width_ry`.
    - No stored band wholly above $\mu+10k_BT$ (`GATE orbital_totals_empty_band`).
    - With `--moments-grid`, a top returned band occupied above $10^{-6}$ on the grid (`GATE htransform_moments_window`).
    - In the standalone command, a QP WFN (`GATE orbmag_qp_wfn`), an `--nbnd` above the bands in `velocity_only.h5`, and on the sweep route no `*.upf` files.

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $u_{n\bk}$, $E_{n\bk}$, $f_{n\bk}$ | cell-periodic state, energy and occupation |
| $\mathbf v_{mn}$ | $\langle m\bk\vert\hat{\mathbf v}\vert n\bk\rangle$, Ry bohr |
| $A_{mn}$ | Berry connection $i\langle u_{m\bk}\vert\nabla_\bk u_{n\bk}\rangle$ |
| $\mathbf m_{n\bk}$, $\boldsymbol\Omega_{n\bk}$ | orbital moment of band $n$ ($\mu_B$); Berry curvature (bohr²) |
| $\mathbf M$, $\mathbf m^{\mathrm{orb}}$ | orbital magnetization; orbital moment per cell ($\mu_B$) |
| $\mu$, $T$, $g(E)$ | chemical potential; Fermi–Dirac temperature; grand-potential weight of (1.5) |
| $\delta$ | degeneracy tolerance, 1.4 meV |
| $\hat{\mathbf n}$, $g_e$ | magnetization axis; electron $g$ factor, 2.00232 |
| $Q_\alpha$, $S$ | $\bk$-independent (Galerkin) basis function; fitted band window |
| $U$, $D_\bk$, $\Delta H$ | DFT-to-QP rotation; covariant derivative; QSGW change of $H$ |

## References {: #refs }

1. T. Thonhauser, D. Ceresoli, D. Vanderbilt and R. Resta, Phys. Rev. Lett. 95, 137205 (2005).
2. D. Xiao, J. Shi and Q. Niu, Phys. Rev. Lett. 95, 137204 (2005).
3. D. Ceresoli, T. Thonhauser, D. Vanderbilt and R. Resta, Phys. Rev. B 74, 024408 (2006).
4. J. Shi, G. Vignale, D. Xiao and Q. Niu, Phys. Rev. Lett. 99, 197202 (2007).
5. D. Xiao, M.-C. Chang and Q. Niu, Rev. Mod. Phys. 82, 1959 (2010).
6. K. Wu, Y. Li, W. Wu, L. Lin, W. Hu and J. Yang, "Hamiltonian transformation for accurate and efficient band structure interpolation", preprint [arXiv:2211.02846](https://arxiv.org/abs/2211.02846).

