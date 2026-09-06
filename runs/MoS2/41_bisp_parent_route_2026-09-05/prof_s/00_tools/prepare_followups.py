"""Prepare immutable follow-up arms from a completed matched donor."""
from pathlib import Path
import shutil

base = Path(__file__).resolve().parent.parent
instrument = (base/'00_tools/profile_driver.py').read_text()
instrument = instrument.replace('    kernel = original_factory(*args, **kwargs)', '''    before, start = snapshot(), time.perf_counter()
    kernel = original_factory(*args, **kwargs)
    receipt('sigma_factory', start, before, dict(options=kwargs))''')
for name, donor in [('14_P_host_boundary','05_P_full_static_baseline'),
                    ('15_F_host_boundary','06_F_full_static_baseline'),
                    ('16_P_weight_shape_ablation','05_P_full_static_baseline'),
                    ('17_P_plan_reuse_ablation','05_P_full_static_baseline')]:
    src, out = base/donor, base/name
    out.mkdir()
    for item in ['cohsex.in','rankwrap.sh','run.sh','inputs.sha256','dipole.h5']:
        shutil.copy2(src/item, out/item)
    shutil.copytree(base/'05_P_full_static_baseline/tmp', out/'tmp')
    (out/'profile_driver.py').write_text(instrument)
    script = 'profile_driver.py' if 'host_boundary' in name else 'ablation.py'
    (out/'driver.sh').write_text((src/'driver.sh').read_text().replace('python3 -u -m gw.gw_jax',f'python3 -u {script}'))
    (out/'driver.sh').chmod(0o755)
    (out/'manifest.yaml').write_text((src/'manifest.yaml').read_text().replace(donor,name)+'instrument: host_boundary_no_nsys_no_hlo_dump\n')
    (out/'cohsex.in').write_text((out/'cohsex.in').read_text().replace(str(src),str(out)))
(base/'13_P_restore_lifetime_ablation/profile_driver.py').write_text(instrument)
