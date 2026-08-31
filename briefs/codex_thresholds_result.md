# Sigma/screening/SC threshold audit — result

| Source threshold | Physical meaning and scale | Degenerate regime | Handling / audit verdict |
|---|---|---|---|
| `src/gw/minimax_screening.py:70`, `x_min = max(cmin - vmax, _TINY)` | Smallest positive Laplace denominator, Ry; set by the excitation support, not the fundamental gap. | A metal or zero/small-gap system makes `cmin-vmax <= 0`, collapsing `x_min` to `1e-12` Ry and requesting an unachievable `R ~= 6.9e12`, scaled error `1e-18`. | **Open defect:** silently floored, then the minimax ladder silently surrenders. The owner-measured Na live-fit evidence is JID 57771859. |
| `services/minimax/src/minimax/door.py:638-640`, rounded cache-key tolerance | Dimensionless requested minimax error; its scale comes from the caller's physical tolerance after nondimensionalisation. | A legitimate scaled tolerance below half of `1e-14` rounded to exactly zero. | **Fixed on the inherited branch:** significant-figure fallback preserves the positive request and non-positive tolerances refuse. |
| `src/gw/sc_iteration.py:1709-1715`, SC `efermi_ev` passed into `compute_sigma_xc` | Current fixed-N chemical potential, eV; it anchors the signed Sigma energy window and is set by the current QP spectrum and occupations. | Reusing the iteration-zero DFT value becomes meaningless after a large SC chemical-potential drift (1.35 eV observed). | **Fixed on the inherited branch:** every iteration passes the map candidate's Fermi level; invalid/non-finite values refuse instead of defaulting. |

Audit status: in progress. The complete literal/derived-threshold census, newly
registered issues, measured fixes, and validation artifacts will replace this note.
