# Retained Na quadrature audit

Diagnostic entry points only. Production drivers, partition/accelerator code,
and compile-agreement policy are unchanged.

`rule_audit.py --source ARM_E/qsgw --output RESULT` reads accepted map 4's
receipt and retained 900-node NPZ. It evaluates the reciprocal on independent
clouds, writes two dense real-time reference rules on the full padded support,
and builds sign-definite replacements on the actual map-4 support with 12 s
and 120 s reduction budgets. The printed maxima are sampled errors, not
analytic sup certificates. Dense-rule truncation has an analytic bound;
quadrature discretization is checked numerically. These are scalar rules,
not delivered Sigma or Hamiltonian errors.

`reconstruct_input.py --source ARM_E/qsgw --output RESULT` uses the existing
EQP parser and the retained rotations to reconstruct an approximate H4. The
saved H4 is explicitly unauthenticated: nine-decimal-place output spectra
cannot recover the exact original input or the occupation bytes bound to W.
Do not pair eqp0_iter0004 with rotation_iter0004 as a restart: eqp0 is F(H4)'s
spectrum while that rotation diagonalizes H4.

`check_acceptance.py current` runs the proposed four-path policy regression
against current production. Expected result: one-shot refuses, SC initial,
SC rebuild and SC disk-cache reuse fail to refuse. `check_acceptance.py strict`
forces the existing sup check on in-process, only for this diagnostic. Expected
result: all four refuse. No production module is changed on disk. Replace
`test_fixed_sc_accepts_the_box_services_finite_fallback` with this refusal
contract when the coordinator approves the production policy.

Run scalar and policy scripts through checkout-local `lx run --jid JOB -N 1 -G 0 -n 1` with
`JAX_PLATFORMS=cpu`. The policy tests deliberately live under the excluded
`bench` directory; the current-policy red test is not added to the default gate.

The coordinator authorized the affine input for a meV diagnostic on September 5:
its residual against retained rotations is 1.4572556481210558e-10 eV. This does
not make it a byte-authenticated restart. `replay_sigma.py` runs the production
SC map at P4 with this input, retained W/head, the geometric bare-X head, and
MP1 occupations at the retained chemical potential. It compares all retained
rules, a 24-node replacement for `ω≥E_F cond:pole_tail`, and an independently
constructed 384-node contour/pane control for that window. All other rules and
the partition are held fixed. Paths and the retained chemical potential are
deliberately specific to this evidence set.

`contour_reference.py` constructs the independent control with composite
Gauss–Legendre quadrature on a decaying complex-time ray. Its truncation bound
is analytic; discretization accuracy is independently sampled. The attempted
`pane_reference.py` fit refused and is retained as evidence, not used in Sigma.
`compare_replay.py` reports covered on-shell samples separately from endpoint
clamping, fixed DFT principal-block eigenvalue shifts, and full-H shifts.

`review_policy.py` probes the exact source of commit `6035f72f` with injected
boundary cases. Its results distinguish the landed finite-error acceptance
policy from remaining nonfinite-value and refusal-message gaps.

Run the combined Sigma leg with `lx run --jid 57930535 -N 1 -G 4 -n 4`,
`LX_BASE_MODULE=lorrax_A`, `LORRAX_CHECKOUT` set to this checkout, and
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. Evidence lives in the sibling sandbox at
`runs/Na/14_quadcheck_2026-09-05`; its `replay/run_replay.sh` records the command.
