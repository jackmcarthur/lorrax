"""Configure the frozen Run300/289 binder for this additive tools-only worktree.

Only the run/source locations and explicit allowed source path list change.
The existing binder independently authenticates every protected Sigma file.
"""
from pathlib import Path
import hashlib

S=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
RUN=S/'runs/frequency_integration_sandbox/304_order_weighted_bt_20260907'
W=Path('/pscratch/sd/j/jackm/wt_sp_order')
OWNER=RUN.parent/'289_na_rpa_reduction_sigma_20260906/prepare_sigma.py'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest()=='e3342e3718de59f47880c8e72776d8ea6c57e43c90c864e28098735e5a14cd2a'
text=OWNER.read_text()
extra=[f'tools/shared_pole_order/{name}.py' for name in
       ('balance','weight','check_gramians','reductions','norm_controls','bind_control','positive_real_audit','validate_models','full_roundtrip','thresholds','local_gauss','shifted_bt','check_shifted','remainder')]
for old,new in (
    ('RUN=Path(__file__).resolve().parent',f'RUN=Path({str(RUN)!r})'),
    ("SOURCE=Path('/global/u2/j/jackm/wt_ff_psiirr_20260905')",f'SOURCE=Path({str(W)!r})'),
    ("ALLOWED=['src/bse/bse_loading.py','src/bse/bse_w_exact.py','src/bse/occupation_pairs.py']",
     "ALLOWED=['src/bse/bse_loading.py','src/bse/bse_w_exact.py','src/bse/occupation_pairs.py']+"+repr(extra)),
):
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(OWNER),'exec'),globals())
