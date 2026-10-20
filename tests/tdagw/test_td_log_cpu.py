"""The run log ``tdagw.out`` of the real TD-aGW driver (CPU host mesh, about 2 min).

``_log_case.py`` writes four decks on a Hamiltonian synthetic window (3v3c,
3x3 k, a full kernel with one stiff outlier quartet) and each runs as
``python -m tdagw.tdgw_jax -i <deck>`` on a 2x2 host mesh (four host
devices), the command a user types.

L1 every log holds its sections in order and closes with the completion line
L2 stdout is a subset of the log: every line of the session's stdout is in it, the
   chunk lines as progress rows with the same samples, time and wall; no log-only
   section reaches stdout
L3 the progress table parses at its fixed columns; rows cover the samples once
L4 the resolved deck lists every key once, default / set / unset as the deck has it;
   keys set but not read are flagged
L5 the invariant drifts are rounding-sized on the Hamiltonian flow (cfm4 < 1e-12;
   dp54: Tr rho exact, Tr rho^2 at the tolerance)
L6 the gate block: the measured wall of one dSigma call, the probes, the chosen dt
   (auto); the split's basis
L7 dp54: no gate, the dSigma wall bounded and not timed
L8 a refused run's log still closes: the gate's line, the refusal message, ERROR
L9 the summary's steady s/fs and its bound on one dSigma call leave out the chunks that
   compiled a program (a new step or sample count, re-derived from the table and dt) and
   are the other chunks' wall over their time and over their dSigma calls
R1 the same checker fails a log missing a section, a row shifted by one column and a
   stdout line the log lacks
"""
import json
import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "src"))

SECTIONS = ("PROVENANCE", "RESOLVED DECK", "SYSTEM", "SELF-ENERGY MODEL",
            "INTEGRATOR AND STABILITY GATE", "PROGRESS", "SUMMARY", "MAJOR-STAGE TIMING",
            "OUTPUT FILES AND INPUTS")
CHUNK = re.compile(r"^  samples +(\d+)- *(\d+)  t = +([0-9.]+) as  wall +([0-9.]+) s$")
NATIVE_MISSING = ("Could not locate liblorrax_ffi", "FfiLibraryNotBuilt")


#: Launcher variables the runtime reads to form a multi-process world; the child driver
#: drops them, so it is one process with four host devices also inside a srun step.
_LAUNCHER = ("SLURM_", "PMI_", "PMIX_", "OMPI_", "PALS_", "JAX_COORDINATOR", "JAX_NUM_PROCESSES",
             "JAX_PROCESS_")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER)}
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    return env


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("tdlog")
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_log_case.py"), str(root)],
                         capture_output=True, text=True, env=_env(), timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    decks = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))[7:])
    res = {}
    for tag, deck in decks.items():
        p = subprocess.run([sys.executable, "-u", "-m", "tdagw.tdgw_jax", "-i", deck],
                           capture_output=True, text=True, env=_env(), timeout=900,
                           cwd=str(root))
        if any(m in p.stderr for m in NATIVE_MISSING):
            pytest.skip("the driver needs the native pair (liblorrax_ffi_host.so); "
                        "run where the module or a built pair provides it")
        log = os.path.join(str(root), f"out_{tag}", "tdagw.out")
        res[tag] = dict(rc=p.returncode, stdout=p.stdout, stderr=p.stderr,
                        log=open(log).read() if os.path.exists(log) else None)
    return res


# ── the checker (R1 runs it on tampered logs) ────────────────────────────────
_FOREIGN = re.compile(r"^(\[|  \[|[WEIF]\d{4} )")   # component tags, runtime receipts, glog


def session_stdout(stdout):
    """The session's stdout lines: the run header through the summary block.

    Lines of other writers that share the stream (native ``[...]`` tags, the
    runtime's exit receipts ``  [compile-cache] ...``, XLA's glog lines) are
    not the driver's and are skipped.
    """
    lines = stdout.splitlines()
    i0 = lines.index("TD-aGW propagation (TDGW conventions)")
    out, in_summary = [], False
    for line in lines[i0:]:
        if _FOREIGN.match(line):
            continue
        if line == "propagation done":
            in_summary = True
        elif in_summary and not line.startswith("  "):
            break
        out.append(line)
    return out


def table_rows(log):
    from tdagw.td_log import parse_progress_row, progress_header
    lines = log.splitlines()
    h = lines.index(progress_header())
    rows = []
    for line in lines[h + 2:]:
        if set(line.strip()) == {"-"}:
            break
        rows.append(parse_progress_row(line))
    return rows


def problems(log, stdout, *, completed=True):
    out = []
    lines = log.splitlines()
    pos = []
    for name in SECTIONS:
        hit = [i for i, l in enumerate(lines[:-1])
               if l == name and lines[i + 1] == "-" * len(name)]
        if not hit:
            out.append(f"section {name} missing")
        else:
            pos.append(hit[0])
    if pos != sorted(pos):
        out.append(f"sections out of order: {pos}")
    status = "COMPLETED" if completed else "ERROR"
    if not any(l.startswith("LORRAX tdagw " + status + " ") for l in lines):
        out.append(f"no '{status}' line")
    try:
        rows = table_rows(log)
    except (ValueError, StopIteration, IndexError) as exc:
        return out + [f"progress table does not parse: {exc}"]
    chunks = []
    have = set(lines)
    for line in session_stdout(stdout):
        m = CHUNK.match(line)
        if m:
            chunks.append(m)
        elif line not in have:
            out.append(f"stdout line not in the log: {line!r}")
    if len(rows) != len(chunks):
        out.append(f"{len(rows)} progress rows for {len(chunks)} chunk lines")
    for r, m in zip(rows, chunks):
        if (r["samples"] != (int(m.group(1)), int(m.group(2)))
                or abs(r["t_fs"] - float(m.group(3)) / 1000.0) > 6e-5
                or abs(r["wall_s"] - float(m.group(4))) > 0.011):
            out.append(f"row {r} does not carry chunk line {m.group(0)!r}")
    for marker in ("RESOLVED DECK", "PROVENANCE", "dPur/Pur", "MAJOR-STAGE TIMING"):
        if marker in stdout:
            out.append(f"log-only text {marker!r} on stdout")
    return out


def _deck_rows(log):
    lines = log.splitlines()
    i = lines.index("RESOLVED DECK") + 3
    rows = {}
    for line in lines[i:]:
        if not line.strip() or re.match(r"^  \d+ keys: ", line):
            break
        key, status = line.split()[:2]
        rows.setdefault(key, []).append((status, line))
    return rows


def _floats(pattern, text):
    return [float(x) for x in re.findall(pattern, text)]


# ── gates ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tag", ["split", "plain", "dp54"])
def test_log_sections_superset_and_table(runs, tag):
    r = runs[tag]
    assert r["rc"] == 0, r["stderr"][-3000:]
    assert r["log"] is not None, "no tdagw.out"
    assert problems(r["log"], r["stdout"]) == []                         # L1-L3
    rows = table_rows(r["log"])
    assert rows[0]["samples"][0] == 0 and rows[-1]["samples"][1] == 23   # 24 samples, once
    assert all(a["samples"][1] + 1 == b["samples"][0] for a, b in zip(rows, rows[1:]))
    assert all(b["dsigma"] >= a["dsigma"] and b["t_fs"] > a["t_fs"] for a, b in zip(rows, rows[1:]))
    prov = r["log"][r["log"].index("PROVENANCE"):r["log"].index("RESOLVED DECK")]
    for field in ("Started        :", "Source         :", "Source closure :", "FFI bundle     :",
                  "\n  JAX ", "\n  mesh "):
        assert field in prov, field


@pytest.mark.parametrize("tag", ["split", "plain", "dp54"])
def test_resolved_deck_marks_defaults(runs, tag):
    from tdagw.td_config import TD_DEFAULTS
    rows = _deck_rows(runs[tag]["log"])
    for key in TD_DEFAULTS:                                              # L4
        if key != "lorrax.trials":
            assert len(rows.get(key, [])) == 1, key
    assert rows["lorrax.method"][0][0] == "set"
    assert rows["lorrax.magnus_frame"][0][0] == "default"
    assert rows["lorrax.stiff_krylov"][0][0] == "default"
    assert rows["control.timesteps.max_step"][0][0] == "unset"
    if tag == "plain":
        assert rows["lorrax.n_corr"][0][0] == "default"
    if tag in ("split", "plain"):
        assert "not used: dp54's step controller" in rows["control.ivp.rtol"][0][1]
    if tag == "dp54":
        assert "not used: magnus2 / cfm4 only" in rows["lorrax.n_corr"][0][1]
        assert "not used" not in rows["control.ivp.rtol"][0][1]


@pytest.mark.parametrize("tag,bound_tr,bound_pur", [("split", 1e-12, 1e-12),
                                                   ("plain", 1e-12, 1e-12),
                                                   ("dp54", 1e-12, 1e-5)])
def test_invariant_drifts_tiny(runs, tag, bound_tr, bound_pur):
    rows = table_rows(runs[tag]["log"])                                  # L5
    assert max(r["d_trace"] for r in rows) < bound_tr
    assert max(r["d_purity"] for r in rows) < bound_pur
    assert "Gamma = 0: both conserved by the exact flow" in runs[tag]["log"]


def test_gate_block_and_split(runs):
    for tag in ("split", "plain"):                                       # L6
        log = runs[tag]["log"]
        wall = _floats(r"one dSigma call : ([0-9.e+-]+) s \(the gate's power iteration", log)
        assert len(wall) == 1 and wall[0] > 0.0, tag
        assert "    sigma         : " in log and "    probe         : dt " in log
        assert "    gate cost     : " in log
    assert "chosen dt     : " in runs["split"]["log"]
    assert "(auto: the largest step that passed)" in runs["split"]["log"]
    assert "  stiff basis     : n_s 4 " in runs["split"]["log"]
    assert "slow map L(1 - Pi) with 4 stiff modes split off" in runs["split"]["log"]
    assert "(the deck's)" in runs["plain"]["log"]


def test_dp54_has_no_gate(runs):
    log = runs["dp54"]["log"]                                            # L7
    assert "  stability gate  : none for dp54" in log
    assert "  one dSigma call : not timed" in log
    assert re.search(r"  dSigma wall     : <= [0-9.e+-]+ s per call", log)


def _compiled_chunks(log, rows):
    """The chunks that build a jitted program, re-derived from the log alone.

    Fixed step: one program per (steps, samples) of a chunk, the steps from the
    chunk's span over the logged dt; dp54: one per (samples, last chunk).
    """
    m = re.search(r"^  step            : dt = ([0-9.e+-]+) as", log, re.M)
    keys, built, t_prev = set(), [], 0.0
    for j, r in enumerate(rows):
        n = r["samples"][1] - r["samples"][0] + 1
        key = ((round((r["t_fs"] - t_prev) * 1000.0 / float(m.group(1))), n) if m
               else (n, j == len(rows) - 1))
        if key not in keys:
            keys.add(key)
            built.append(j)
        t_prev = r["t_fs"]
    return built


@pytest.mark.parametrize("tag", ["split", "plain", "dp54"])
def test_steady_rates_leave_out_compiling_chunks(runs, tag):
    from tdagw.td_log import _chunks
    log = runs[tag]["log"]                                               # L9
    rows = table_rows(log)
    built = _compiled_chunks(log, rows)
    steady = [j for j in range(len(rows)) if j not in built]
    assert steady, (tag, built)
    m = re.search(r"  steady s/fs     : ([0-9.e+-]+) s per fs \((chunks? [0-9, -]+); "
                  r"(chunks? [0-9, -]+) compiled", log)
    assert m and m.group(2) == _chunks(steady) and m.group(3) == _chunks(built), (tag, built)
    d = {k: sum(rows[j][k] - rows[j - 1][k] for j in steady) for k in ("wall_s", "t_fs", "dsigma")}
    n = len(steady)                  # the table's walls carry 0.005 s, its times 5e-5 fs
    rate = d["wall_s"] / d["t_fs"]
    assert abs(float(m.group(1)) - rate) <= 1e-3 * rate + (0.011 + 1e-4 * rate) * n / d["t_fs"]
    got = _floats(r"(?:bound it by|dSigma wall     :) <= ([0-9.e+-]+) s", log)
    want = d["wall_s"] / d["dsigma"]
    assert len(got) == 1 and abs(got[0] - want) <= 1e-3 * want + 0.011 * n / d["dsigma"], \
        (tag, got, want)


def test_refused_run_log_closes(runs):
    r = runs["refused"]                                                  # L8
    assert r["rc"] != 0 and r["log"] is not None
    log = r["log"]
    assert "INTEGRATOR AND STABILITY GATE (THE RUN STOPPED HERE)" in log
    assert "\nREFUSAL\n-------\n  ValueError: GATE td_magnus_stability" in log
    assert re.search(r"\nERROR    .* \[td_magnus_stability\]\n(?:         .*\n)*"
                     r"         ValueError; the full message is on stderr\n", log)
    assert re.search(r"\nLORRAX tdagw ERROR [^\n]* · [^\n]*\n", log)
    stability = [l for l in r["stdout"].splitlines() if l.startswith("  stability       : ")]
    assert stability and stability[0] in log


def test_checker_can_fail(runs):
    r = runs["plain"]                                                    # R1
    log, stdout = r["log"], r["stdout"]
    assert problems(log, stdout) == []
    no_summary = log.replace("\nSUMMARY\n-------\n", "\n")
    assert any("SUMMARY" in p for p in problems(no_summary, stdout))
    lines = log.splitlines()
    from tdagw.td_log import progress_header
    i = lines.index(progress_header()) + 2
    shifted = "\n".join(lines[:i] + [" " + lines[i][:-1]] + lines[i + 1:])
    assert problems(shifted, stdout), "a row shifted by one column parsed"
    extra = stdout.replace("propagation done", "propagation done\n  an extra stdout line")
    assert any("not in the log" in p for p in problems(log, extra))
