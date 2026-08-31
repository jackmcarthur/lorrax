# LOEWNER-SI result — heavy lane

**Numbers first.** The matched P=4 Si census covered 294,912 matrix elements at each order. Conditions are for the realization actually inverted (smallest retained singular value after the current method's truncation).

| poles | legacy condition p50 / p99 / max | current condition p50 / p99 / max | max reduction |
|---:|---:|---:|---:|
| 8 | 2.685370e9 / 3.016000e10 / 3.068307e11 | 2.685370e9 / 3.016000e10 / 3.068307e11 (**bit-identical**) | 1.00x |
| 10 | 1.665582e10 / 2.352579e11 / 9.393570e12 | 1.203143e9 / 1.734676e10 / 8.620667e11 | 10.90x |
| 12 | 1.271524e12 / 1.176389e13 / 4.905191e14 | 9.661149e10 / 8.387882e11 / 9.951109e12 | 49.29x |

The 8-pole result is bit-identical on gapped Si too: poles, residues, condition, backward error, and valid-pole counts all match over all 294,912 elements on all four ranks. At 12 poles, 4,060 legacy elements exceeded `1/rcond = 1.0e13`; current has zero. The actual 12-pole store is complete and finalized: observed condition `9.951109e12` (only 0.489% below the certificate), backward error `9.95876e-18` versus `1.49012e-8` allowed, and `mpa_fit_complete = TRUE`. Thus it is finalizable/consumable, but with thin condition headroom.

## Reconstruction against the sampled W

Normalization is `RMS(|model-W|)/max(|W|)` and `max(|model-W|)/max(|W|)` over the whole four-rank field.

| poles | relative RMS, legacy -> current | relative max, legacy -> current | absolute max, legacy -> current |
|---:|---:|---:|---:|
| 8 | 2.502403829e-6 -> 2.502403829e-6 | 3.746707347e-4 -> 3.746707347e-4 | 12125.87918 -> 12125.87918 |
| 10 | 1.394891380e-6 -> 1.394890442e-6 | 1.586957177e-4 -> 1.586948722e-4 | 5136.043255 -> 5136.015891 |
| 12 | 1.061314413e-6 -> 1.061049894e-6 | 9.909323461e-5 -> 9.907487479e-5 | 3207.062841 -> 3206.468642 |

Conditioning therefore improves the 12-pole aggregate fit slightly (relative RMS by 0.0249%, relative max by 0.0185%), is numerically neutral at 10 poles, and exactly neutral at 8. It is not a uniform elementwise improvement: the 12-pole valid-count decision changed for 360 elements, and some changed elements improved while others worsened.

**Owner's one-line answer:** this pencil method is **just better, not conditioning-optimal**—it performs one row-norm pass followed by one column-norm pass, with no optimization or nontrivial lower-bound certificate; establishing optimality requires certified lower/upper bounds for `min cond2(D_r L D_c)` on every positive diagonal-equivalence class (and, for broader realization optimality, comparison across sample partitions/realizations), whereas the only bound established here is the trivial `cond2 >= 1`.

## Evidence

- Completed source Si run: four A100s, four ranks, 2x2 mesh, gapped (`0.67666 eV` DFT; `0.97057 eV` effective-H), 80.09 s driver wall; sample store SHA-256 `de3a0d801e41ed671eaf8bd96294a7b27a561dbc8462ea8e7b9a449632419238`.
- Census: JID `57789884.37`, P=4, four A100s, 95 s, all ranks proved source `4bb9e725451be6552f7ff77a246731d84c5e6272`. Timing is diagnostic only because the replay did not stamp an allocator. Aggregate SHA-256 `433f78c6b49742da7815370942e00fc5f040d17655cfcdc76f2d52ee60278965`.
- CPU gate: **134 passed** in 148.10 s. Evidence directory: `/pscratch/sd/j/jackm/wt_loewner_si_2026-08-31/briefs/loewner_si_evidence`.
- Two earlier launcher attempts entered no census work (missing JAX, then Python 3.6); their logs are preserved beside the useful leg.

Branch `audit/loewner-si-2026-08-31`; pushed. No shared allocation was released.
