# The MPA compute mode: what exists, what refuses, what lands next

> **This page predates the working method and is kept for the compute-mode
> reasoning only. For what MPA is and how the landed code works, read
> [the method guide](../mpa_method_guide.md).**

This is the page the driver points at when you set `compute_mode = mpa` and
it stops. It exists so that the refusal is a pointer to an explanation
rather than a dead end.

## What `compute_mode = mpa` means

`mpa` is the multipole approximation to the screened interaction: instead of
representing W's frequency dependence with a single plasmon pole fitted to
two samples — which is what `gn_ppm` and `hl_ppm` do — it fits n complex
poles (Ω_p, B_p) to W sampled on a double-parallel grid in the complex
frequency plane, and evaluates Σ_c(ω) from those poles. The owner's
shorthand for this work is "FF", for full frequency, because the point of
it is to stop approximating the frequency integral with one pole; the deck
key is spelled `mpa` because every other value on this axis names an
ansatz, and "full frequency" names a family of ansätze rather than one of
them. That reasoning is written out in full in the `ComputeMode` docstring
in `src/gw/gw_config.py`, together with the alternative that was rejected.

## Why the mode exists before the kernels do

The mode is declared on the axis, parses from a deck, has a row in the
channel-availability table, and is named explicitly at every site in the
tree that dispatches on `compute_mode` — and it refuses to run.

That ordering is deliberate. A compute mode is dangerous in exactly the
window between "someone can select it" and "everything that branches on it
knows about it", because during that window a dispatch site with an `else`
branch will quietly serve the new mode whatever the old modes got. The Σ
dispatch had precisely that shape: anything that was neither bare exchange
nor COHSEX fell into the two-point plasmon-pole pipeline. An MPA run
landing there would have fitted a GN pole to two W samples and reported the
result as its Σ_c(ω), and no stage downstream could have told. Closing that
window before any kernel exists is cheaper than closing it afterwards, and
it is testable today.

So what landed with the mode is the safety skeleton: the value, the parser,
the refusal, the exhaustive dispatch, and the channel table. What did not
land is any physics.

## What refuses, and where

Selecting `mpa` stops the run at driver entry — before the wavefunction
read, before ISDF, before any allocation is spent — with a message naming
the mode. The refusal is `NotImplementedError`, which is a different
exception from the `ValueError` a misspelled mode value gets from the
parser, because those are different operator mistakes: `compute_mode = mpaa`
means "no such mode", `compute_mode = mpa` means "that mode, not yet".

The entry refusal is a courtesy, not the safety mechanism. Every downstream
site refuses independently, so a caller that bypasses the driver — the
self-consistency loop, a test, a future entry point — cannot reach a kernel
that would misinterpret the mode:

- `gw.screening.screening_requests_for` has an `mpa` branch that refuses
  rather than returning the PPM's `{0, probe}` pair. MPA's W is sampled on
  the double-parallel grid, and that grid is not two points.
- `gw.sigma_dispatch.compute_sigma_xc` refuses any mode with no plasmon-pole
  model before it reaches the PPM pipeline, so the `else` that used to
  absorb new modes is gone.
- `gw.ppm_pipeline.compute_ppm_sigma_pipeline` refuses a non-pole-model mode
  at its own entry, which is where the "HL, or else GN" reading of the mode
  lives.
- `gw.gw_output.persist_w0_and_head` refuses to stamp a `{0, probe}`
  head grid onto a restart file for a mode whose W was never evaluated
  there.

Deleting the mode's row from `UNIMPLEMENTED_MODES` in `gw_config.py` is the
gesture that turns the mode on. The suite that pins these refusals fails
loudly the moment that row goes, which is the intended forcing function:
whoever lands the fit stage has to replace each refusal with the real
behaviour rather than discover later that one of them was still standing.

## What the fit stage will build

The theory is settled and written up in `~/MPA_THEORY_PLAN.md`; the parts
that bear on this mode's shape are summarised here so the skeleton's names
can be read against them.

The fit stage produces, per (q, μν) column, a set of complex poles Ω_p with
residues B_p, fitted from W evaluated on the double-parallel sample grid —
two horizontal lines in the complex-ω plane, with a semi-homogeneous
powers-of-two real partition nested in the pole count. The infrastructure
for this is already in the tree under `src/gw/mpa/`: the sample grid
(`sampling`), the sampling object and its plans including `mpa_plan`
(`sample_plan`), the normalised Padé-in-z² solve with its ordered guards
(`pade_fit`), the conditioning and held-out harness (`diagnostics`), the
complex-frequency W evaluator (`evaluator`), the walk over the (q, ν-column)
grid (`tiling`), the driver that composes them (`fit_driver`), and the
staged B/Ω store (`file_io.mpa_store`). None of it is wired to a compute
mode yet, which is the gap this mode's refusal marks.

The Σ stage then consumes those complex poles. The four-branch
decomposition survives the move from real to complex poles under a single
time-ordered continuation; the crossing core targets the full complex
resolvent rather than a sine-only kernel, because the split Im-channel
projection was an economy specific to the Hybertsen-Louie plasmon pole and
computes the wrong analytic object here. Poles are summed before the
spatial FFT — a design that dispatches a separate spatial kernel per pole
is rejected outright — and the accumulation runs as a 14-pass structure.
Pole widths select quadrature rules by octave bucket, but the exact Γ_p is
never rounded and no pole is dropped or evaluated outside a certified
envelope.

The channel table already records the consequence for the outputs: `mpa`
builds the same two Σ channels the PPM modes build — bare exchange Σ_x and
dynamic correlation Σ_c(ω) — by a different producer. That is exactly why
the table alone could not be the safety net, and why the mode also refuses:
two modes can be indistinguishable in what they produce and completely
different in how, and only the second difference is the dangerous one.

## Where the rest is written down

- `src/gw/gw_config.py` — the `ComputeMode` docstring (the naming decision
  and the rejected alternative), `MODE_SIGMA_CHANNELS` (the
  channel-availability table and the rule for adding a mode),
  `UNIMPLEMENTED_MODES` and `refuse_unimplemented_compute_mode`.
- `src/gw/mpa/__init__.py` — what each module of the staged MPA
  infrastructure owns, and the note that its location is a parking spot
  rather than a ruling.
- `tests/test_ff_compute_mode.py` — the refusals, the exhaustiveness of
  every mode-dispatch site, and the table's completeness ratchet.
- `~/MPA_THEORY_PLAN.md` — the theory plan itself: the fit stage, the Σ
  stage, the tabulation campaign, the error budget and the ranked risks.
