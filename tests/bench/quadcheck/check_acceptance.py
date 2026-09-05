"""Run the policy test against current SC and an in-process strict proposal."""
import sys
import runtime
runtime.bootstrap(platform='cpu')
import pytest
import gw.sigma_box_plan as planner
if sys.argv[1]=='strict':
    original=planner._fit_rule
    def strict(*args,**kwargs):
        kwargs['enforce_sup_error']=True
        return original(*args,**kwargs)
    planner._fit_rule=strict
raise SystemExit(pytest.main([str(__file__).replace('check_acceptance.py','test_acceptance_proposal.py'),'-q','--confcutdir='+str(__file__).rsplit('/',1)[0]]))
