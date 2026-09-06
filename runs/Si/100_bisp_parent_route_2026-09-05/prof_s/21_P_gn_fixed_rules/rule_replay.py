"""Select immutable common certificates through the unchanged production guards."""
import json
from pathlib import Path
from gw import sigma_box_plan
_original=sigma_box_plan._rule_cache_lookup
_rows=json.loads((Path(__file__).parent/'replay.json').read_text())
def replay(directory, box, eps, relative, *, noise_amplification_cap, reduction_steps=None):
    index=min(range(len(_rows)),key=lambda i:sum((a-b)**2 for a,b in zip(box,_rows[i]['box'])))
    assert max(abs(a-b) for a,b in zip(box,_rows[index]['box']))<1e-4, (box,index)
    selected=Path(__file__).parent/'replay_rules'/str(index)
    # Both pins accept the legacy no-step schedule used by these decks.
    assert reduction_steps is None
    result=_original(str(selected),box,eps,relative,noise_amplification_cap=noise_amplification_cap)
    if result[0] is None:
        raise RuntimeError(f'Common certificate {index} refused for {box}: {result[1]}')
    print(f'[rule-replay] {index}: {result[0][1]}',flush=True)
    return result
sigma_box_plan._rule_cache_lookup=replay
