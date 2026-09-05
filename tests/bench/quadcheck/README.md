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

Run all scripts through checkout-local `lx run --jid JOB -N 1 -G 0 -n 1` with
`JAX_PLATFORMS=cpu`. The policy tests deliberately live under the excluded
`bench` directory; the current-policy red test is not added to the default gate.
Actual Sigma replay remains a P4 requirement and needs an authenticated input
H/E/U/occupation checkpoint plus W/head and the retained rule session.
