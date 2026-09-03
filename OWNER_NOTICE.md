# Owner notice (2026-09-03, after your launch) — read before implementing EvaluationPolicy

The `fermi` policy's window membership is decided PER BAND, per iteration: band m is
inside the window iff its current QP energy E_mk lies in [omega_min, omega_max] for
EVERY k. Never a per-(m, k) test. A pair with both bands inside is evaluated at
(E_mk, E_nk) at each k; otherwise at (E_F, E_F) at every k. The design doc
(`docs/dev/notes/DESIGN_self_consistent_loop.md`, commit on this branch) is updated;
`git pull` brings it. The `clamp` policy is unchanged (no membership). Include in the
study a count of band flips per iteration for `fermi` (bands whose membership changed),
since that is the mechanism the owner expects this rule to calm.

Decision rule (owner, 2026-09-03): `fermi` is the primary policy. If it does not converge on the study arms (or converges to a worse window-consistency than `clamp`), revert to the literature method = `clamp` (QSGW mode A at the energies, edge-clamped) as the shipped behaviour without waiting for a ruling; report both either way.

# Owner authorization (2026-09-03 09:51)
More GPU concurrency is authorized: run independent arms/legs simultaneously on separate nodes, and use interactive or PREMIUM QOS allocations (see `runs/Na/03_soc48b_rebuild_2026-09-01/run_control_p16.sh` request_alloc for the premium salloc pattern) when the shared interactive pool is short. Still one four-rank leg per four-GPU node, never two legs on one node (the concurrent-launch race is registered), never sbatch.

# Owner view on the scissor (2026-09-03 09:53)
The owner considers a per-k scissor the smarter form in principle and expects it not to matter much in typical systems. Pass 3 keeps the constant Δ as the study baseline (the per-k form drove MoS2 to 0/6 in pass 2); if the constant converges, add ONE per-k arm on MoS2 P=30 as a comparison and report the difference, but do not make per-k the default on this evidence.
