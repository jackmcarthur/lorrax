"""Prepare copied P16 controls and one-warm-TT-unit native captures."""
from pathlib import Path
import subprocess
import sys
r=Path(__file__).resolve().parent.parent
prep=Path(__file__).with_name('prepare_scan.py')
for name,baseline,profile in [('19_P16_baseline',True,False),('20_P16_baseline_profile',True,True),('21_P16_candidate',False,False),('22_P16_candidate_profile',False,True)]:
 t=r/name
 subprocess.run([sys.executable,str(prep),str(r/'01_P_static_baseline'),str(t)],check=True)
 p=t/'run.sh';p.write_text(p.read_text().replace('-N 1 -G 4 -n 4','-N 4 -G 4 -n 16'))
 p=t/'manifest.yaml';p.write_text(p.read_text().replace('geometry: P4 one rank per GPU','geometry: P16 one rank per GPU')+'\nsource_mode: '+('orchestrator71ae0bde exact src/services' if baseline else 'committed scan and class producer')+'\n')
 if profile:
  driver=(r/'03_P_static_profile/driver.sh').read_text().replace('diff --exit-code 71ae0bde -- src services tests','diff HEAD -- src services tests')
  text=(r/('03_P_static_profile' if baseline else '15_P_restore_profile')/'profile_driver.py').read_text()
  label='sigma_block' if baseline else 'sigma_class'
  index=21 if baseline else 7
  text=text.replace('def measured(function, label, metadata):', 'kernel_calls = 0\n\ndef measured(function, label, metadata):')
  text=text.replace('    def call(*args, **kwargs):\n        before, start',f'''    def call(*args, **kwargs):
        global kernel_calls
        selected = label == '{label}' and kernel_calls == {index}
        if label == '{label}':
            kernel_calls += 1
        if selected:
            assert ctypes.CDLL('libcudart.so').cudaProfilerStart() == 0
        before, start''')
  text=text.replace('        receipt(label, start, before, metadata)', '''        if selected:
            jax.effects_barrier()
            assert ctypes.CDLL('libcudart.so').cudaProfilerStop() == 0
        receipt(label, start, before, metadata)''')
  text=text[:text.index('def captured_main():')]+'''def captured_main():
    return gw_jax.main()

run_main_and_finalize(captured_main)
'''
  (t/'profile_driver.py').write_text(text)
 else:
  driver=(t/'driver.sh').read_text()
 pin='71ae0bde' if baseline else 'HEAD'
 guard=f'git -C {r.parents[3]} diff --exit-code {pin} -- src services > source_pin.diff\n'
 driver=driver.replace('export LORRAX_DEBUG_PRINT=1\n',guard+'export LORRAX_DEBUG_PRINT=1\n')
 (t/'driver.sh').write_text(driver)
 (t/'driver.sh').chmod(0o755)
