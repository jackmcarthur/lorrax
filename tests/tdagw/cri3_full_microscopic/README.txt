CrI3 516 full-microscopic campaign instrumentation -- PREPARED, NOT RUN

These helpers preserve the 20 October 2026 matched-trajectory plan. They
do not establish that a native material pole model, complex self-energy,
Hamiltonian fixed point, or 10 fs trajectory has passed. See the current
numerical evidence in docs/theory/td_agw_dynamical.md before launching.
Do not substitute the earlier spatially projected B/C trajectories.

prepare.py writes new source-hashed manifests and native TD decks. Supply
the actual source deck, 516/07 bundle, new output directory, and complete
immutable SOURCE TREE hash, not a commit hash. Use --launch-root and
--launch-source when preparing locally for another machine. Preparation
never submits a job or authorizes the proposed time/storage ceilings.

After a full native screening/Sigma proof is admitted, run_matrix.py can
run one explicitly selected manifest tag collectively inside a bounded
P4 payload. It refuses a source/tree mismatch and existing outputs. The
observer records current U/E/full rho, complex Sigma branch tables and
Gamma/D/MH, physical health and cost forecasts. No clipping or number
repair is performed. A failed or cropped leg is not a completed 10 fs run.

Default controls are 2v2c, all 978 centroid ports/36 q/160 virtual bands,
raw retarded-reference collision and hHF+MH, dt25 as, samples50 as,
refresh0.5 fs, pulse center3 fs/width1 fs/energy0.07 Ha, fields0/7e9/1e10
V/m. These are exploratory numerical settings, not convergence results.
The prepared DFT density is a declared quench. B/C share the same t0
construction; B freezes W afterward, C updates it. Their A static TD-aGW
reference has a different coherent functional and alone cannot isolate
collision damping. The matched B/C comparison isolates W updates only
subject to the documented numerical and statistical approximations.

plot_full_native.py requires actual native authenticated B/C outputs and
matching time/field/k/weight data. It refuses missing or nonmatching data.
Explicit --allow-partial crops to checkpoint-written samples and labels
partial coverage. It plots polarization, nonlinear field normalization,
carrier/trace/purity/Pauli health and optional actual material gap/mixing.
It does not invent data or infer damping from unfinished calculations.

Example preparation (replace every angle-bracket argument):
 python prepare.py --out <NEW_DIRECTORY> --source <SOURCE_ETA010.IN> \
   --ref <516_07_BUNDLE> --source-tree <40_HEX_TREE>
Collective execution, only after admission and resource approval:
 python -u run_matrix.py <MANIFEST.JSON> --tags E0_joint
Read-only plotting:
 python plot_full_native.py <MANIFEST.JSON> --a-root <STATIC_A_ROOT> \
   --out <NEW_PLOT_DIRECTORY>

Focused CPU test_helpers.py checks instrumentation, generated native deck
contracts and partial/authenticated plot reading with tiny synthetic data.
It is not a dynamics or GW validation. Run it explicitly from the repo root
with the matching source closure and four CPU devices as other TD gates.

analyze_general_pencil.py is a bounded offline numerical diagnostic, not
a model-admission tool. It studies the saved q0 descriptor at relative
rank tolerances1e-8/1e-9/1e-10/1e-11 without changing production settings.
Preferred input includes E/A, canonical poles/maps, frequency-only
GC=C^H C and GB=B B^H, and two probe_mixed_J=C^H W_exact B^H matrices
with probe_sites_ry and probe_exact_norm_sq. No microscopic mu axes are
gathered for these metrics. Without them only descriptor-coordinate
conditioning, not physical pole observability, can be diagnosed.

 python analyze_general_pencil.py --input <CAPTURE.NPZ> \
   --output <NEW_REPORT.JSON> --max-order <RECORDED_CAP> \
   --max-workspace-gib 4

Apply an external time bound appropriate to the recorded pencil size.
The helper refuses an existing output or a workspace overrun; candidate
orders are diagnostic only and must still pass exact full-W/Sigma checks.
Gram norm differences retain raw squared norms and roundoff bounds because
small defects may be cancellation limited. Its tiny tests demonstrate a
possible unobservable high-growth pole and validate physical Gram metrics;
they do not diagnose the material refusal without the actual capture.
