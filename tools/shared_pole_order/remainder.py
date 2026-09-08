"""Route SHIFT's exact custom-model three-node remainder through this checkout."""
import importlib.util
import json
from pathlib import Path
import sys
from balance import S,sha

PATH=S/'runs/DEV/152_shared_pole_push_2026-09-07/exchange/shift/remainder_model.py'
EXPECTED='9030144db61f8a74e31342673314621e9619614440990682a820f8d9f2dc8863'


def main():
    assert sha(PATH)==EXPECTED
    source=PATH.read_text()
    old="'--nodes','2'";assert source.count(old)==1
    source=source.replace(old,"'--nodes','3'")
    namespace=dict(__name__='order_shift_adapter',__file__=str(PATH))
    exec(compile(source,str(PATH)+':three-node-adapter','exec'),namespace)
    original_load=namespace['load']
    def load(name,path):
        if name=='shift_binder':path=Path(__file__).with_name('bind_control.py')
        module=original_load(name,path)
        if name=='custom_signed_owner':module.load=load
        return module
    namespace['load']=load
    output=Path(sys.argv[sys.argv.index('--output')+1])
    namespace['main']()
    (output/'ORDER_ADAPTER.json').write_text(json.dumps(dict(adapter_path=str(Path(__file__).resolve()),adapter_sha256=sha(__file__),
        delegate_path=str(PATH),delegate_sha256=EXPECTED,nodes=3,
        changes=['delegate CLI nodes2 to nodes3','same source-binding owner configured for clean ORDER checkout'],
        sign='PLUS-MINUS is parent-minus-candidate correction'),indent=2)+'\n')


if __name__=='__main__':main()
