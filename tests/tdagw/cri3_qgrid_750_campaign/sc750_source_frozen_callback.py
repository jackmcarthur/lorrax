"""Verification-only SOURCE-FROZEN W with the native current full-rho Sigma.

This is neither production joint_frozen initialization nor matched B/C
evidence. Only the W provider changes: a validated positive equilibrium source
bank is converted exactly to signed ordered poles. The existing production
evaluator retains current U/E/rho, active44 rates, full750 virtual MH, the
authenticated spectral tail and head, and the existing anchored HF solver.
"""
from pathlib import Path
import hashlib
import importlib.util
import time


ROLE = "SOURCE_FROZEN_DIAGNOSTIC"
DIRECT_REQUEST = dict(q=0, z_ry=[-0.16154070956060437, 1.0066523909325316])
DIRECT_STOP = "GATE shared_general_dielectric_diagnostic_stop:"


def read_one_site_witness(root, request, logical_dimension):
    """Read durable scalar witnesses without promoting inverse bounds to admission."""
    import json
    records = {}
    for kind in ("basic", "inverse"):
        path = Path(root)/f"q_000_dielectric_00_{kind}.json"
        row = json.loads(path.read_text())
        if (row.get("q") != 0 or row.get("z_ry") != request["z_ry"]
                or row.get("admitted") is not False or row.get("sigma_computed") is not False
                or row.get("logical_dimension") != logical_dimension):
            raise ValueError("GATE source_frozen_dielectric_witness: mismatched/unadmitted scalar artifact")
        records[kind] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inverse_bounds=row.get("inverse_bounds"), inverse_error=row.get("inverse_error"))
    return dict(status="UNADMITTED_DIRECT_DIELECTRIC_RECORDED", request=request,
        artifacts=records, physical_root_claim=False, note="Read inverse error/bounds explicitly; no automatic resolution claim")


def bind_source_frozen_evaluator(source, prepared, options, *, model, meta, original_factory,
                                 direct_probe=None, probe_root=None, verify_probe=None):
    """Bind the existing native evaluator to one explicitly frozen W handle.

    Parameters are native bundles; no spatial/device arrays are copied here.
    The ordinary factory captures its provider during construction. Restoring
    the module immediately leaves production dispatch unchanged elsewhere.
    """
    from gw import shared_pole_screening as screening
    if (options.get("screening") != "frozen" or options.get("b_lo") != 100
            or options.get("n_active") != 44 or prepared["nb"] != 750):
        raise ValueError("GATE source_frozen_callback: explicit frozen W, active44/full750 required")
    cache = {}
    original_provider = screening.screen_general_shared_poles

    def source_provider(*args, **kwargs):
        if direct_probe is not None and "probe" not in cache:
            from gw.ppm_sigma import fresh_shared_pole_capacity
            # A controlled diagnostic stop skips the ordinary builder cleanup.
            # Its ledger must not contaminate the subsequent Sigma admission.
            probe_args = list(args)
            probe_args[2] = fresh_shared_pole_capacity(args[2], prepared["mesh"])
            probe_kw = dict(kwargs, snapshot=dict(kwargs["snapshot"],
                direct_dielectric_probe=dict(direct_probe)), run_dir=str(Path(probe_root).parent),
                label="one_site_epsilon_suspect")
            try:
                original_provider(*probe_args, **probe_kw)
            except RuntimeError as error:
                if not str(error).startswith(DIRECT_STOP):
                    raise
            else:
                raise ValueError("GATE source_frozen_dielectric_witness: expected named unadmitted stop")
            cache["probe"] = verify_probe()
        if "model" not in cache:
            cache["model"] = model() if callable(model) else model
        ordered = cache["model"]
        receipt = dict(ordered["source_frozen_receipt"], diagnostic_role=ROLE,
            screening_control=ROLE, refreshed_screening=False, matched_BC_admission=False,
            source_W_unshifted=True, response_lag_gamma_applied_to_W_ev=0.0,
            current_U_E_fullrho_lines=True, spectral_tail="production native source owner",
            one_site_dielectric=cache.get("probe"))
        # Equilibrium in this diagnostic names the raw collision reference,
        # not a new W calculation or a claim of collision stationarity.
        return dict(shared_pole=ordered, meta=meta, receipt=receipt)

    screening.screen_general_shared_poles = source_provider
    try:
        return original_factory(source, prepared, options)
    finally:
        screening.screen_general_shared_poles = original_provider


def run_td_probe(source, *, out, deck_path, budget_s=900., storage_gib=64.):
    """Run the existing .6fs zero owner with a separately labelled W control."""
    from common.collectives import rank0_transaction, rank0_read
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.shared_pole_reference import source_frozen_ordered_model
    from gw.shared_pole_screening import _shared_pole_tables
    from tdagw import snapshot_runtime as sr
    from tdagw.td_config import read_td_deck
    import json

    spec = importlib.util.spec_from_file_location(
        "sc750_source_frozen_common", Path(__file__).with_name("sc750_td_callback.py"))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    deck = read_td_deck(str(deck_path))
    common.validate_probe_deck(deck)
    active = common.probe_active_window(deck)
    if (active.start, active.stop) != (100, 144) or deck.values["lorrax.collision_screening"] != "joint_frozen":
        raise ValueError("GATE source_frozen_callback: explicit zero44 frozen deck required")
    if not source["config"].write_poles:
        raise ValueError("GATE source_frozen_callback: private write_poles=true artifact policy required")
    positive = source.get("W_by_role", {}).get("shared_pole")
    if not positive or source.get("spectral_tail_reference") is None or source.get("head") is None:
        raise ValueError("GATE source_frozen_callback: authenticated native source W/tail/head required")
    if source.get("diagnostic_role") is not None:
        raise ValueError("GATE source_frozen_callback: nested diagnostic substitution forbidden")
    common.source_response_policy(source)
    mesh = source["mesh"]
    meta = fresh_shared_pole_capacity(source["meta"], mesh)
    tables = _shared_pole_tables(meta, meta.q_sym, meta.mu_basis.canonical_indices)
    bank_root = Path(out).resolve().with_name(Path(out).name+"_SOURCE_FROZEN_W")
    def model_provider():
        started = time.perf_counter()
        ordered = source_frozen_ordered_model(positive, meta, mesh_xy=mesh, tables=tables,
            path=str(bank_root/"model.h5"))
        report = dict(diagnostic_role=ROLE, status="EXACT_SOURCE_ADAPTER_COMPLETE_NOT_BC_ADMISSION",
        refreshed_screening=False, matched_BC_admission=False,
        source_W_unshifted=True, response_lag_gamma_applied_to_W_ev=0.0,
        configured_response_gamma_ev=float(deck.values["lorrax.collision_adiabatic_ev"]),
        adapter=ordered["source_frozen_receipt"], adapter_seconds=time.perf_counter()-started,
        source_frame=source["frame"], active=[100, 144], virtual_bands=750,
        source_input_sha256=hashlib.sha256(Path(source["input_deck"]).read_bytes()).hexdigest(),
        callback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tail_provenance={k:v for k,v in source["spectral_tail_reference"].items() if k != "fit"},
        head_scope="unchanged actual native source correlation head; fine bare HF once",
        energy_unit="Ry", current_lines="production block-embedded instantaneous U/E/fullrho",
            resource_scope="native adapter capacity owns bank; common budgets own TD outputs; launcher caps total process")
        rank0_transaction(bank_root/"receipt.json", stage="source_frozen_callback/receipt",
            write=lambda:(bank_root/"receipt.json").write_text(json.dumps(report, indent=2, default=str)+"\n"))
        return ordered
    original_factory = sr.make_native_snapshot_evaluator
    probe_root = bank_root/"one_site_epsilon_suspect_general_shared_pole"

    def verify_probe():
        return rank0_read(probe_root, stage="source_frozen_callback/dielectric_witness",
            read=lambda:read_one_site_witness(probe_root, DIRECT_REQUEST, int(meta.n_rmu)))

    def factory(current_source, prepared, options):
        return bind_source_frozen_evaluator(current_source, prepared, options, model=model_provider,
            meta=meta, original_factory=original_factory, direct_probe=DIRECT_REQUEST,
            probe_root=probe_root, verify_probe=verify_probe)

    sr.make_native_snapshot_evaluator = factory
    try:
        return common.run_td_probe(source, out=out, deck_path=deck_path, budget_s=budget_s,
            storage_gib=storage_gib, diagnostic_role=ROLE)
    finally:
        sr.make_native_snapshot_evaluator = original_factory
