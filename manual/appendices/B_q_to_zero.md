# Appendix B — The q → 0 head of W

At $\mathbf q \to 0$, $\mathbf G = \mathbf G' = 0$, the screened interaction's
head is the product of a $1/q^2$ divergence and a $q^2$ zero, so the value of
that one matrix slot is a limit, not a number the body solve can supply. The
bare side is the mini-BZ average $\bar v_0$ of §6.2. The polarizability side
vanishes as $\chi_{00}(\mathbf q) = q_a S_{ab} q_b + O(q^4)$, with the
Cartesian $S$ tensor assembled from `dipole.h5` transition matrix elements
(§9.1) and angle-averaged over the mini-BZ.

Because the $q\to0$ matrix elements are band-diagonal in the charge channel,
the head enters $\Sigma$ analytically per state rather than through the
$(\mu,\nu)$ convolution — which is why it survives as a separate object at all,
and why the transverse channels of Chapter 8 cannot use the same trick (their
structure factor $j^i_{mn}$ is not band-diagonal, so their head is a Cartesian
tensor that must be injected into the tile).

**Do not look for the equations here.** The single owner for the
$q\to0$ treatment of every Lorentz channel — the $S(\omega)$ definition, the
Schur fold against the Γ wings and the body ($S^{\rm eff}$), the static COHSEX
shifts, the GN/HL single-pole head fit, the MPA complex-pole head, the bare TT
tensor head, and the packed static photon head with its Hall term and its
declared omissions — is
[`docs/theory/four-current-head-corrections.md`](../../docs/theory/four-current-head-corrections.md).
The producers and their shardings are in
`docs/architecture/four_current_wiring.md`. The S-tensor sign and index
convention is `docs/theory/s-tensor-convention.md`.

## Deck keys, in one place

| key | what it decides |
|---|---|
| `head_correction` | `full` (Schur fold applied) \| `no_local_fields` (direct $\epsilon$ head, diagnostic) \| `off` (no Γ contribution — debug only; see the 2026-09-01 ruling in `docs/architecture/decisions.md`) |
| `wcoul0_source` | `s_tensor` (default, from `dipole.h5`) \| `epshead` (import a static BGW `epsmat` head, for parity work) |
| `vhead`, `whead_0freq`, `whead_imfreq` | explicit overrides of the bare head and of $W_h$ at $\omega=0$ / $i\omega_p$. The override fires only when both the bare value and the frequency-matched $W$ are set; setting one and omitting the other is warned about, not silently half-applied |
| `ppm_head_omega_h_ry` | pin $\Omega_h$ directly instead of fitting it |
| `head_minibz_average`, `mc_average_placement` | the analytic-sphere / widened-Voronoi fold for $q=0$, and where the mini-BZ average is applied at $q \ne 0$ |
| `bispinor_tt_head_correction` | the bare transverse tensor head of Chapter 8 |

In 2D the truncated kernel diverges only as $1/q$ and the same construction
applies with the slab kernel's angular structure. In 3D the $1/q^2$ part uses
the Baldereschi–Tosatti analytic sphere. Box truncation (`sys_dim = 0`) has no
divergence to repair, and the transverse head is refused there rather than
silently zeroed.
