# AMENDMENT — `K^d_B` UNDER ζ SHARDING: REGISTERED AND STRUCK (2026-08-08)

**`tests/test_bse_coupling_zeta_sharding.py` is registered here and struck in
the same commit.**  Registering a defect nobody had ever seen fail needs a
word of explanation: the non-TDA coupling block's screened-direct term
`K^d_B` was **wrong at every P > 1** and had been since it was written, and it
was not on this list because **no test in the tree applied the coupling block
on a mesh with more than one device**.  The existing non-TDA gates all run on
the 1×1 mesh `lx test`'s conftest pins, which is exactly the one configuration
where the defect is invisible.  A row that only appears once someone writes
the missing check is still a row: it is registered with its measured red, and
struck with the fix, so the census carries the fingerprint rather than losing
it to a green file.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56522011), 4×A100, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| tree | `/pscratch/sd/j/jackm/kdb_0808/wt`, branch `fix/kdb-zeta-sharding-2026-08-08` off `main` @ `28ff477f` |
| files | `src/bse/bse_ring_comm.py` (the coupling encode), `tests/test_bse_coupling_zeta_sharding.py` (new) |
| prose record | `~/lorrax_bse_perf_2026-08-08/FIX_kdb_sharding.md` |

**THE DEFECT — a ζ shard carried away by a `ppermute`, not a conjugation.**
The coupling encode carries Henneke's `j_c ↔ j_v` swap, so the ζ index it
builds from the conduction axis (sharded on `'x'`) is `ν` (sharded on `'y'`)
and the one it builds from the valence axis (on `'y'`) is `μ` (on `'x'`) — the
pairing is CROSSED relative to the resonant block's, where the ζ index a stage
produces lands on the same mesh axis as the orbital axis it consumed.  The
shipped chain therefore ring-rotated a partially-contracted intermediate along
`'y'`, the very axis its `ν` shard lived on.  A `ppermute` moves **every** axis
of the buffer it is handed, so each `'y'` rank accumulated its neighbours' ζ
tiles against its own ζ shard.  At P=1 a `ppermute` is the identity, so the
term was bit-exact there and nowhere else.

**MEASURED, on the record deck (Si 4×4×4 SOC, `nontda/deck_clean`, N=1024),
same payload, same process, 1×1 vs 2×2:**

| quantity | pre-fix | post-fix |
|---|---:|---:|
| `K^d_B` ‖P1−P4‖/‖P1‖ | **5.525e-01** | **1.749e-15** |
| `K^d_B` ‖K−Kᵀ‖/‖K‖ at 2×2 | **6.911e-01** | **2.864e-11** (= its P=1 value) |
| coupling correction, state 1 | **−0.223277 meV** | **−0.697956 meV** |
| max\|Im λ\| of the 2048-dim operator | 2.647e-06 Ry | 1.025e-13 Ry |
| `A` (resonant), `K^x_B` (coupling exchange) | clean | **bit-identical to pre-fix** |

The physics cost was two thirds of the coupling correction, lost silently, on
exactly the configuration multi-process non-TDA exists to run.

**THE FALSE CASE SHIPS WITH THE CHECK.**  `test_the_pre_fix_coupling_encode_is_caught`
monkeypatches the 2026-08-08 chain back in and requires both gates to fire —
cross-mesh `‖P1−P4‖/‖P1‖` ≥ 1e-3 and `‖K−Kᵀ‖/‖K‖` ≥ 1e-3 — **and** requires
the twin to leave `A` and `K^x_B` bit-clean, so the red proves the gate is
pointed at the coupling screened-direct term and not at the mesh in general.
The gate is portable: synthetic payload, CPU host devices, no GPU, no restart
file, no deck.

**Struck on:** the new file green (6 passed) with its red twin red; the deck
detector green at 2×2 on both encode routes (`low_mem` ring and `all_gather`);
P=1 **bit-identical** to pre-fix on all four blocks (`0.000e+00`, exact array
equality), which is the acceptance that says the fix touched only the P>1
path; and the coupling correction at 2×2 reproducing the P=1 value to
**0.0000 µeV** over all twenty states.
