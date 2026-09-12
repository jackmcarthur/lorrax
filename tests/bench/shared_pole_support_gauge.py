"""Probe basis covariance of shared-pole normalization on a planted measure.

The scalar-normalization arm executes the canonical reducer with exactly its
``scale`` assignment replaced by one trace-based scale per support. This
test-only AST substitution is authenticated below; no production code changes.
Run on a compute node with JAX_PLATFORMS=cpu.
"""


def main():
    import argparse
    import ast
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import subprocess

    import runtime
    runtime.bootstrap(platform="cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gw import shared_pole_constructor as owner
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(owner.__file__).resolve()
    checkout = source.parents[2]
    seed = 20260910
    rng = np.random.default_rng(seed)
    n, latent_n, width = 8, 72, 4
    c = rng.normal(size=(n, latent_n)) * .04
    poles = np.geomspace(.08, 6., latent_n)
    points = (-.20, -.22, -.25, -.32, -.50, -1., -2.)
    m1 = c @ c.T / 2
    m3 = (c * poles) @ c.T / 2
    adj = lambda a: a.conj().swapaxes(-1, -2)

    def sample(s):
        return (c / (s-poles)) @ c.T, (-c / (s-poles)**2) @ c.T

    ports = [np.linalg.eigh(-sample(s)[0])[1][:, -width:] for s in points]
    ports.append(np.linalg.eigh(m1)[1][:, -width:])
    block_widths = tuple(q.shape[-1] for q in ports)
    offsets = np.cumsum((0, *block_widths))
    r = int(offsets[-1])

    def block_scales(g, active):
        diagonal = jnp.real(jnp.diagonal(g, axis1=-2, axis2=-1))
        parts = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            mean = jnp.sum(diagonal[:, lo:hi], axis=-1) / (hi-lo)
            parts.append(jnp.broadcast_to(1/jnp.sqrt(mean[:, None]),
                                          (g.shape[0], hi-lo)))
        return jnp.concatenate(parts, axis=-1) * active

    reducer_source = inspect.getsource(owner.reduce_shared_pole_pencil)
    tree = ast.parse(reducer_source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == "scale"]
    assert len(matches) == 1, "canonical reducer no longer has one normalization assignment"
    matches[0].value = ast.parse("_support_scales(g, active_columns)", mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = dict(owner.__dict__, _support_scales=block_scales)
    exec(compile(tree, str(source) + ":scalar_support_probe", "exec"), namespace)
    reducers = {"column": owner.reduce_shared_pole_pencil,
                "support_scalar": namespace["reduce_shared_pole_pencil"]}

    def mm(a, b, transa="N", transb="N"):
        return (adj(a) if transa == "C" else a) @ (adj(b) if transb == "C" else b)

    def batch(a):
        return jnp.asarray(a[None], dtype=jnp.complex128)

    rows, stored, references = [], {}, {}
    held = (-.11, -.37, -1.4, .65+.3j, 2.4+.5j)
    for gauge in range(7):
        rotated = []
        span_error = 0.
        for q in ports:
            u = np.eye(q.shape[1]) if gauge == 0 else np.linalg.qr(
                rng.normal(size=(q.shape[1], q.shape[1])))[0]
            q_new = q @ u
            span_error = max(span_error, float(np.linalg.norm(q_new @ q_new.T - q @ q.T)))
            rotated.append(q_new)
        states, latent = [], []
        for s, q in zip(points, rotated[:-1]):
            w, dw = sample(s)
            states.append((s, batch(q), batch(w @ q), batch(dw @ q)))
            latent.append((c.T @ q)/(s-poles)[:, None])
        qi = rotated[-1]
        infinity = tuple(batch(a) for a in (qi, m1 @ qi, m3 @ qi))
        latent.append(c.T @ qi)
        x = np.concatenate(latent, axis=-1)
        exact_g = adj(x) @ x
        pencil = owner.assemble_shared_pole_pencil(states, infinity, matmul=mm)
        assembly_error = float(np.linalg.norm(np.asarray(pencil[0])[0]-exact_g)
                               / np.linalg.norm(exact_g))
        for mode, reducer in reducers.items():
            model, diag, coefficients = reducer(
                pencil, jnp.ones((1, r), dtype=bool),
                eigh=jnp.linalg.eigh, matmul=mm, gates=gates)
            b, t, active = [np.asarray(a)[0] for a in model]
            y = np.asarray(coefficients)[0]
            values = np.asarray([(b/(s-t)) @ adj(b) for s in held])
            latent_basis = x @ y[:, active]
            projector = latent_basis @ adj(latent_basis)
            moments = np.asarray([b @ adj(b), (b*t) @ adj(b)])
            spectrum = np.asarray(diag["gram_spectrum_relative"])[0]
            if gauge == 0:
                references[mode] = values, projector, moments
            ref_w, ref_p, ref_m = references[mode]
            relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
            row = dict(mode=mode, gauge=gauge, retained_rank=int(active.sum()),
                       pencil_side=r, active_cut=bool(0 < active.sum() < r),
                       physical_support_projector_max_fro=span_error,
                       gram_assembly_relative=assembly_error,
                       gram_min_relative=float(spectrum.min()),
                       gram_condition=float(np.asarray(diag["gram_condition"])[0]),
                       gram_valid=bool(np.asarray(diag["gram_valid"])[0]),
                       metric_valid=bool(np.asarray(diag["retained_metric_positive"])[0]),
                       metric_relative=float(np.asarray(diag["retained_metric_relative"])[0]),
                       minimum_active_pole=float(t[active].min()),
                       last_kept_relative=float(spectrum[spectrum > 1e-8].min()),
                       first_dropped_relative=float(spectrum[spectrum <= 1e-8].max()),
                       held_model_max_relative=max(relative(a, b) for a, b in zip(values, ref_w)),
                       latent_projector_relative=relative(projector, ref_p),
                       m1_relative=relative(moments[0], ref_m[0]),
                       m3_relative=relative(moments[1], ref_m[1]),
                       held_error_vs_exact_max_relative=max(relative(a, sample(s)[0])
                                                            for a, s in zip(values, held)),
                       normalized_gram_spectrum=spectrum.tolist())
            rows.append(row)
            stored[f"{mode}_{gauge}_w"] = values
            stored[f"{mode}_{gauge}_b"] = b
            stored[f"{mode}_{gauge}_poles"] = t
            print(json.dumps({k: v for k, v in row.items() if k != "normalized_gram_spectrum"}), flush=True)

    assert all(row["physical_support_projector_max_fro"] < 1e-13 for row in rows)
    assert all(row["gram_assembly_relative"] < 1e-12 for row in rows)
    assert all(row["active_cut"] and row["gram_valid"] and row["metric_valid"] for row in rows)
    assert all(row["minimum_active_pole"] > 0 for row in rows)
    maxima = {mode: max(row["held_model_max_relative"] for row in rows if row["mode"] == mode)
              for mode in reducers}
    result = dict(status="PASS", scope="Planted real resolvent, exact support-span rotations; no Si attribution or QP accuracy claim",
                  job_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
                  constructor_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  canonical_reducer_sha256=hashlib.sha256(reducer_source.encode()).hexdigest(),
                  scalar_override="Exactly one AST assignment: scale = _support_scales(g, active_columns)",
                  seed=seed, n=n, latent_n=latent_n, points=points, block_widths=block_widths,
                  jax_version=jax.__version__, backend=jax.default_backend(),
                  gauge_family="Identity plus six seeded real orthogonal rotations in every support block",
                  cutoff=1e-8, maximum_held_model_relative_change=maxima, rows=rows)
    np.savez(args.output / "models.npz", **stored)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
