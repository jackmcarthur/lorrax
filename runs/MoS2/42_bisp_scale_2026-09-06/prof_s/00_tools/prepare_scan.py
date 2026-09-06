"""Prepare a copied six-by-six scan leg with explicit runtime provenance."""
from pathlib import Path
import argparse
import shutil

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('target', type=Path)
p.add_argument('--profile', action='store_true')
a = p.parse_args()
s, t = a.source.resolve(), a.target.resolve()
w = Path(__file__).resolve().parents[5]
t.mkdir()
for name in ('cohsex.in', 'rankwrap.sh', 'run.sh'):
    shutil.copy2(s/name, t/name)
for name in ('WFN.h5', 'kin_ion.h5', 'dipole.h5', 'centroids_charge.txt', 'centroids_current.txt'):
    (t/name).symlink_to((s/name).resolve())
shutil.copytree(s/'tmp', t/'tmp')
(t/'cohsex.in').write_text((t/'cohsex.in').read_text().replace(str(s), str(t)))
(t/'manifest.yaml').write_text(f'''run_id: {t.name}
system: MoS2
pipeline: gwjax
platform: perlmutter
variant_of: {s}
reused: copied tmp zeta and rule caches; immutable WFN, centroids, kin_ion, dipole
source: source_head.txt plus source.diff
pool: 57982945
geometry: P4 one rank per GPU
steps:
  sigma:
    state: pending
''')
driver = (s/'driver.sh').read_text()
driver = driver.replace('diff --exit-code 71ae0bde', 'diff HEAD')
if a.profile:
    driver = (t.parent/'03_P_static_profile/driver.sh').read_text().replace('diff --exit-code 71ae0bde', 'diff HEAD')
    shutil.copy2(t.parent/'03_P_static_profile/profile_driver.py',t/'profile_driver.py')
(t/'driver.sh').write_text(driver)
(t/'driver.sh').chmod(0o755)
print(t)
