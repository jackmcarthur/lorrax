# AWDEBUG finding — `write_w` is a debug feature; BSE is owed nothing new

Branch `lane/sp-awdebug-2026-09-11`, base `e769b4fe` (`lane/sp-afixes-2026-09-11`),
which carries the AWRITE export repair. Owner ruling 2026-09-11: *"make writing
all of them a debug feature turned off by default."*

## What changed

`write_w` was already `default false`. The ruling is about its *status*, so the
change makes the status structural rather than a comment.

1. **`write_w` moves into the existing `DebugConfig` group.** It is read as
   `config.debug.write_w`; there is no longer a top-level `config.write_w` to
   mistake for a production output. `write_poles` stays exactly where it was,
   top-level beside `write_restart_tensors`.
2. **A parse-time `WARNING -- DEBUG` deck report** fires when the key is true,
   through `_print_deck_report` — the same rank-0 reporter the retired-key
   report uses, in `_resolve_shared_pole_inputs`, which runs inside
   `read_lorrax_input`. It names the cost and the reason BSE does not need it.
3. **Documentation** (`docs/input_reference.md`, the `DebugConfig` field, the
   `_DEFAULTS` note and the `export_shared_pole_outputs` docstring) says plainly:
   this writes the full sample bank for debugging; BSE's static W does not need
   it, because `W_c(0) = -b Λ⁻¹ b†` comes from `write_poles`.

Files: `src/gw/gw_config.py`, `src/gw/shared_pole_screening.py`,
`src/file_io/shared_pole_store.py`, `docs/input_reference.md`, plus three test
files.

## The mechanism I reused, and why

**`DebugConfig` is the repo's existing debug-key mechanism**, and the repo says
so in its own words. `_DEFAULTS["write_qsgw_datasets"]` carries the line

> `NOT A DEBUG FLAG, which is why it is here and not in `debug`.`

That is a rule stated at the one place it is decided: a debug-only flag belongs
in the `debug` group, a supported output does not. `sigma_freq_debug_output` is
the flag-shaped precedent inside that group (a gated debug dump, same shape as
this one); `write_wfn_h5` is the output-shaped one. `write_w` is a gated debug
dump, so it goes there.

**The `WARNING -- DEBUG` token is the repo's existing loud-debug wording** —
`gw_config.incumbent_bispinor_head_record`, `gw/w_isdf.py:2339`,
`gw/mpa/sigma.py:598`, `gw/minimax_screening.py:433`. `_print_deck_report` is
the existing parse-time rank-0 reporter. Nothing new was invented: **no new deck
key, no env var, no cache, no fast path.**

I considered and rejected two alternatives. A `LORRAX_DEBUG_*` env var would be
a new dial for a key that already exists and already parses. Renaming the key
(`debug_write_w`) would break every deck that names it and buys nothing the
group membership does not already buy.

## The BSE answer, with numbers

**No BSE consumer reads the sample bank. None reads the pole export either.**
`grep` over `src/bse/` and `src/bandstructure/` for `shared_pole`, `_w.h5`,
`_poles.h5`, `read_shared_pole_bank` and `validate_shared_pole_model` returns
zero hits. What BSE actually reads for static W is `W0_qmunu` in
`tmp/isdf_tensors_*.h5`, gated on the `W0_ready` attr
(`src/bse/bse_loading.py:1055`, `:1408`; `src/bse/vq_interp.py:508`).

**Two facts follow, and the second is a real gap that `write_w` does not close.**

*First*, turning `write_w` off costs BSE nothing, because nothing read it.

*Second*, on the decks these keys are accepted on — `compute_mode = mpa` with
`sigma_w_model = shared_pole`, enforced at `gw_config.py:2884` — **the driver
never writes `W0_qmunu` at all.** `gw.screening.driver_persists_w0` returns
`False` for `ComputeMode.MPA` (`src/gw/screening.py:1045`), because
`persist_w0_and_head`'s `{0, probe}` head grid is not MPA's sample set. So an
MPA restart file carries the all-zeros placeholder with `W0_ready = False`, and
a BSE run against it takes the bare-V fallback with its banner
(`bse_loading.py:60`). That gap is not one the frequency bank could ever have
filled: **the bank has no ω = 0 sample.** Its line ladder starts at `E = 0`
(`shared_pole_recipe.py:634`, `low = [i*low_step for i in range(...)]`) at
height `h = 4η` (`:621`, `height_eta_factor`), so the first line sample is
`W(4iη)`, not `W(0)`.

**The only exact route to ω = 0 is the model, and `write_poles` already exports
it.** Verified at `src/gw/shared_pole_constructor.py:1016-1021`:

```python
s = _sample_point(recipe, int(sample_id)) ** 2
weights = jnp.where(mask, 1 / (s-poles), 0)
value = mm(b * weight[:, None, :], b, transb="C")
```

so `W_c(s) = b diag(1/(s-Λ)) b†` with `s = z²`, hence `W_c(0) = -b Λ⁻¹ b†`
exactly, from `physical_factor` (`b`) and `poles2_ry2` (`Λ`) — both already in
`<map>_poles.h5`. Static screened `W(0) = v + W_c(0)`.

### Cost of a materialised static `W`, both reference decks

`16·n²` bytes per q (complex128). Deck geometry from claim 2162: si_p4
`n = 368`, 8 raw parents, 64 q in the full BZ, `Kmax = 2010`; na_p16 `n = 896`,
29 parents, 512 q, `Kmax = 1589`.

| | si_p4 | na_p16 |
|---|---:|---:|
| static `W(0)`, per q | 2,166,784 B (2.07 MiB) | 12,845,056 B (12.25 MiB) |
| static `W(0)`, on the 8/29 parents | 17,334,272 B (16.53 MiB) | 372,506,624 B (355.25 MiB) |
| static `W(0)`, full BZ (what BSE reads) | 138,674,176 B (132.25 MiB) | 6,576,668,672 B (6.125 GiB) |
| `write_poles` compact payload | 94,807,744 B (90.42 MiB) | 660,986,096 B (630.37 MiB) |
| `write_poles` file as written | 178,660,510 B (170.4 MiB) | 1,315,176,119 B (1.225 GiB) |
| `write_w` bank file as written | 1,469,737,888 B (1.369 GiB) | 20,868,709,551 B (19.436 GiB) |
| bank / poles, actual files | 8.23x | 15.87x |

The parent-set ratio `staticW / poles` is exactly `n/Kmax` — 0.183 on Si, 0.564
on Na — because `b` is `n×K` and `W` is `n×n`. Both decks fit `K > n`, so a
static `W` on the parents is *smaller* than the pole export. That is the wrong
comparison, for two reasons: the pole file is written anyway (Σ and any-frequency
`W` need it, a static `W` serves neither), so a static-W key is **additional**
bytes; and BSE's reader wants the **full BZ**, where the same object is **1.46x**
(Si) and **9.95x** (Na) the pole payload.

### Deriving on load versus storing

Reconstruction is a column scale by `-1/Λ_j` then one `n×K · K×n` GEMM per
parent — `n²K` complex MACs:

| | per parent | over all parents |
|---|---:|---:|
| si_p4 | 2.72e8 MACs, 2.18 GFLOP | 17.4 GFLOP |
| na_p16 | 1.28e9 MACs, 10.21 GFLOP | 296.0 GFLOP |

Na's 296 GFLOP is milliseconds of fp64 arithmetic on a 16-GPU pool. Against it,
just *reading* the 6.125 GiB full-BZ static `W` back costs ~2.1 s at the 2.919
GiB/s figure this tree already quotes for restart I/O (`restart_q_storage`, in
`docs/input_reference.md`), on top of writing it. The parents unfold to the full
BZ through `symmetry_maps.unfold_isdf_operator`, the same service the RPA path
and the restart readers already use.

**Recommendation: add no third key.** `write_poles` is complete for BSE's static
`W` and is 10x smaller than the full-BZ object it generates. If a BSE consumer
is to be wired to the shared-pole export, the work is a reader that evaluates
`v - b Λ⁻¹ b†` on load and unfolds — not a new output. The separate, real gap is
that `compute_mode = mpa` writes no `W0_qmunu` at all; closing it is a decision
about `persist_w0_and_head`'s head grid on MPA, and it belongs to whoever owns
that refusal. **Not implemented here**, per the lane brief.

## Verification, and its scope

Pool 58212398, attached with `--jid`; **this lane allocated nothing**.

| gate | scope | result |
|---|---|---|
| P4 export format fixture | `tests/test_shared_pole_outputs.py` `__main__`, 4 ranks. Writes both exports through the production writers **with `write_w` on** and compares every canonical payload byte — `factor`, `poles2_ry2`, `K`; `Wc`, `dWc_ds`, `M1`, `M3`, `z_ry`, `distinct_id` — against the source stores, plus the verbatim WFN `mf_header`, the canonical centroid coordinates and the `b`↔`factor` hard link | **PASS**, job.step **58212398.35**, receipt `runs/frequency_integration_sandbox/367_awdebug_20260911/01_p4_format/receipt.json` |
| Deck parsing | 8 config/deck suites incl. `test_shared_pole_inputs`, `test_afixes_review`, `test_deck_dials`, `test_deck_doctor_config`, `test_qp_solver_config` | **313 passed**, 7.26 s, pool 58212398 |
| New parse-time test | `test_write_w_is_a_debug_key_announced_at_parse_time`: the key still parses, still defaults off, has no top-level attribute, and the `WARNING -- DEBUG` report fires at parse time naming `write_poles` — while `write_poles = true` alone stays silent | **PASS** (in the 313) |
| `tools/gate0.sh` | AST suites `test_layering` / `test_crossfile_requests` / `test_env_registry` | **pass**; the `rules_gate` findings and CLAIMS-lint rows it also reports are pre-existing and name no file this lane touched |

One deselected failure, `test_mpa_sampling_config.py::test_uncertified_smearing_family_refuses_by_name`.
**Pre-existing on the base commit** — verified by stashing this lane's entire
diff and rerunning the single test, where it fails identically. Stale regex
against the `occ_smearing_family` message, unrelated to this change; registered
in the sandbox `KNOWN_LORRAX_ISSUES.md`.

## What I am not claiming

- **No production driver leg was run.** No Si or Na GW run was executed on this
  branch, in either arm. The byte-identity evidence is the P4 format fixture,
  not a production deck; the production byte-identity evidence in the record is
  AWRITE's (58166522.3 / .7) and it stands on the base commit this branch starts
  from.
- **No Σ measurement.** The change touches no numerical path: with the key off
  the same boolean is evaluated from a different dataclass field, and no
  arithmetic, sharding or ordering moved. That is an argument from the diff, not
  a measured Σ comparison, and it is not a claim of bit-identical Σ — ruling 65d
  forbids requiring that anyway.
- **No timing claim.** No band was taken, no `placement_audit.py` was run,
  nothing here is a performance statement. AWRITE's measured 924.62 s Na
  output-enabled driver wall stands as the cost of turning the debug path on.
- **No BSE run.** The BSE conclusion is read from source — consumer greps,
  `driver_persists_w0`, the `W0_ready` gates and the recipe's line grid — and
  from claim 2162's measured geometry. No BSE leg was executed to confirm that
  an MPA-parent restart takes the bare-V fallback; the code path is explicit
  (`bse_loading.py:60`) but I did not exercise it.
- The reconstruction FLOP figures are arithmetic counts from deck geometry, not
  timed kernels.
