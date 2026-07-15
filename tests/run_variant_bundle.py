"""Run several ``gw.gw_jax`` invocations in ONE process (Tier-2 speedup).

Each Tier-2 from-restart variant used to be its own subprocess, paying
python+jax import (~4-5 s) and a full retrace of the pipeline graph
(grid-independent, CPU-bound) per launch — measured 8-14 s/variant vs
2-9 s of actual compute (reports/suite_speedup_2026-07-15/).  This runner
amortizes import + tracing across the whole bundle: variants share jitted
kernels wherever shapes match (all restart variants do).

Safe because: ``init_jax_distributed()`` runs once at ``gw.gw_jax`` import;
the LORRAX_* debug knobs are read at call time (not import time); and every
module-level cache in the pipeline holds jitted functions or pure math
results keyed on their full inputs — no run-scoped data (audited 2026-07-15).

Spec JSON:  {"variants": [{"name", "run_dir", "input_name", "env": {K: V}}]}
            env values of null UNSET the variable for that variant.
Results JSON: [{"name", "status": "ok"|..., "error"}] — one entry per
variant, always written; a crashed variant does not stop the bundle.
Per-variant driver stdout goes to <run_dir>/variant_stdout.txt.
"""

import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    spec_path, results_path = sys.argv[1], sys.argv[2]
    spec = json.loads(Path(spec_path).read_text())

    import gw.gw_jax as gw_jax  # module import initializes jax/distributed

    results = []
    for var in spec["variants"]:
        saved = {}
        for k, v in (var.get("env") or {}).items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        cwd0 = os.getcwd()
        buf = io.StringIO()
        status, err = "ok", ""
        try:
            os.chdir(var["run_dir"])
            with contextlib.redirect_stdout(buf):
                rc = gw_jax.main(["-i", var["input_name"]])
            if rc not in (0, None):
                status = f"returncode={rc}"
        except SystemExit as e:  # argparse / driver sys.exit
            if e.code not in (0, None):
                status, err = f"exit={e.code}", buf.getvalue()[-2000:]
        except BaseException:
            status, err = "error", traceback.format_exc()
        finally:
            os.chdir(cwd0)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        Path(var["run_dir"], "variant_stdout.txt").write_text(buf.getvalue())
        results.append({"name": var["name"], "status": status, "error": err})
        print(f"[bundle] {var['name']}: {status}", flush=True)

    Path(results_path).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
