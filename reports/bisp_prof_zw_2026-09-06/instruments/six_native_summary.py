"""Aggregate native profiler kernel receipts without equating nested ranges with wall time."""
from pathlib import Path
import csv,json
from prepare_6x6 import root
summary={}
for name in ('49_P6_fresh_units','50_F6_fresh_units','61_P6_chi_before_units','65_P6_class_units','72_P6_streamed_units','75_F6_chi_units'):
 r=root/name; classes={}; ranges=[]
 for row in csv.DictReader((r/'stats_cuda_gpu_kern_sum.csv').open()):
  label=row['Name'].lower()
  kind=('collective service (NCCL)' if 'nccl' in label else 'GEMM' if 'gemm' in label else 'FFT and normalization' if 'fft' in label or 'lrx_scale' in label else 'transpose fusion' if 'transpose' in label else 'dynamic update' if 'dynamic_update' in label else 'other fusion/kernel')
  item=classes.setdefault(kind,{'kernel_ms':0,'launches':0});item['kernel_ms']+=int(row['Total Time (ns)'])/1e6;item['launches']+=int(row['Instances'])
 for row in csv.DictReader((r/'stats_nvtx_gpu_proj_sum.csv').open()):
  ranges.append({'range':row['Range'],'projected_ms':int(row['Total Proj Time (ns)'])/1e6,'instances':int(row['Range Instances']),'gpu_ops':int(row['Total GPU Ops'])})
 summary[name]={'kernel_classes':classes,'ranges':ranges}
Path('six_analysis/native_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,data in summary.items():
 print(name,json.dumps(data['kernel_classes']))
 print('key ranges',json.dumps([x for x in data['ranges'] if 'XlaModule' in x['range'] or ('ffi_call' in x['range'] and 'Gv_Gc' in x['range']) or 'dot' in x['range']]))
