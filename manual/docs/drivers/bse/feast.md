---
driver: bse
page: theory
---

# The FEAST route

`bse.bse_feast` The contour-projection eigensolver of `bse.bse_jax`, kept off the main BSE pages.
{: .page-summary }

This page is unlisted. It collects what the BSE pages used to say about the FEAST route, which the owner removed from the visible manual on 2026-10-21. Since 2026-10-21 the route runs only with `feast = true`; before that it was the command-line default. Without `full_bse = false` it solves the full problem, and with the default RPA kernel $D + w_x V$ it never builds $W$. It prints Ritz values only and writes no file. It refuses unless every in-window Ritz pair has a residual below `feast_tol_mev` (`GATE feast_converged`). A FEAST flag without `feast = true` refuses.

## Overview {: #overview }

!!! treatment in-progress "feast · Contour projection in KPM windows · feast = true"
    A KPM density of states splits the spectrum into windows of equal state count; FEAST projects each window by shifted GMRES at contour nodes and prints Ritz values per window. Missing: eigenvector output, `eqp_corrections` (refused on this route), and user windows (`feast_window1/2` have no effect).

    ```bash
    # deck keys in the -i deck: feast = true
    python -m bse.bse_jax -i gw.in
    ```


## Method {: #method }

**Contour projection** <span class="status in-progress">In progress</span>

For each KPM window $[E_a, E_b]$, a trial block is filtered by a contour integral of the resolvent, then Rayleigh–Ritz.

Each resolvent is applied by shifted GMRES (`gmres_max_iter`, `gmres_tol`) at `feast_n_quad1` then `feast_n_quad2` nodes, two FEAST iterations. With `full_bse = true` the conjugate nodes are added.

$$
Q = \oint \frac{dz}{2\pi i}(z - H)^{-1} Y \approx \sum_j w_j (z_j - H)^{-1} Y, \qquad Q^\dagger H Q\, y = \Omega\, Q^\dagger Q\, y \tag{4.2}
$$

Subspace
:   contour-filtered block, per window

```bash
# deck keys in the -i deck: feast = true
python -m bse.bse_jax -i gw.in
```


<figure class="flowchart" markdown>
[![BSE windowed solve and the absorption spectrum](../../assets/figures/b3b.png)](../../assets/figures/b3b.png)
<figcaption><strong>Fig. B3b.</strong> Energy windows from the KPM density of states (stage 1), the FEAST contour projection per window (stage 2), oscillator strengths by the Haydock recursion or by overlap with eigenvectors (stage 3), and the spectrum, which both routes give as the same Lorentzian sum (stage 4).</figcaption>
</figure>

## Keywords {: #keywords }

Optional
{: .kw-group }

### `feast = true` {: .kw data-type="flag" data-default="off" }
Selects the FEAST route. It cannot be combined with `the eigensolve` or `kpm_dos = true`.

### `feast_tol_mev` {: .kw data-type="float" data-default="0.01" }
Every in-window Ritz pair's residual $\lVert Hx - \Omega x\rVert$ must be below this, in meV, before any value is reported (`GATE feast_converged`). Out-of-window pairs are leakage and are not tested.

### `feast_iter` {: .kw data-type="int" data-default="2" }
FEAST subspace iterations.

### `feast_n_lanczos` {: .kw data-type="int" data-default="10" }
FEAST route. Minimum Lanczos steps for the spectral bounds; the bound Lanczos runs adaptively up to 50.

### `feast_buffer` {: .kw data-type="float" data-default="0.05" }
FEAST route. Fractional buffer on the upper spectral bound.

### `feast_n_quad1`, `feast_n_quad2` {: .kw data-type="int" data-default="4, 8" }
FEAST route. Contour nodes in the first FEAST iteration and in later ones.

### `feast_quadrature` {: .kw data-type="enum" data-default="ellipse" }
FEAST route. Contour rule: trapezoid on an ellipse of aspect 0.2, or a Zolotarev rational filter.

`ellipse` `zolotarev`
{: .kw-values }

### `feast_ritz_count` {: .kw data-type="int" data-default="4" }
FEAST route. Ritz values reported per window.

### `gmres_max_iter`, `gmres_tol` {: .kw data-type="int, float" data-default="10, 1e-2" }
FEAST route. Shifted-GMRES iterations and relative tolerance per contour node.

### `gmres_seed` {: .kw data-type="int" data-default="0" }
FEAST route. Seed of the start block and of the KPM window estimate.

### `gmres_fp32` {: .kw data-type="flag" data-default="off" }
FEAST route. Shifted solves in single precision.

### `feast_window1`, `feast_window2` {: .kw data-type="eV pair" data-default="—" }
FEAST route. Accepted, without effect: the driver always derives the windows from KPM.

### `feast_units_ev_per_ry` {: .kw data-type="float" data-default="13.6056980659" }
FEAST route. Ry-to-eV factor of the report.

