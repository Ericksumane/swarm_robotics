"""Predictive gate is a veto, preserves yaw and never accesses true peer state."""
import copy
import numpy as np
from swarm_core.predictive_stop import PredictiveStopGate
gate=PredictiveStopGate(dict(horizon_s=1.,prediction_step_s=.1,clearance_margin_m=.03,history_bound_scale_m=1.))
local=np.array([.2,0.,.3,.15,.8,.3,.8,.25,1.]+[0.]*7)
rows=np.zeros((2,11));rows[0,:]=[.7,0.,0.,-1.,.2,0.,0.,.7,-.3,1.,.02];rows[1,:]=[0.,3.,0.,1.,0.,0.,0.,3.,2.,1.,.02]
o=dict(local_features=local,formation_history=dict(neighbor_features=rows,neighbor_mask=np.ones(2)))
action=np.array([.7,.1]);result,guarded=gate.apply(action,o)
assert guarded and result[0]==0 and np.isclose(result[1],action[1])
receding=copy.deepcopy(o);receding['formation_history']['neighbor_features'][0,:6]=[1.,0.,0.,1.,.3,0.]
result,guarded=gate.apply(action,receding);assert not guarded;np.testing.assert_allclose(result,action)
missing=copy.deepcopy(o);missing['formation_history']['neighbor_mask'][0]=0
assert gate.apply(action,missing)[1]
bad=copy.deepcopy(o);bad['local_features'][0]=.300001
result,guarded=gate.apply(action,bad);assert guarded and not result.any()
print('Received-only predictive veto, unchanged yaw, missing peers and out-of-profile speed: PASSED')

# Progress-oriented configuration is explicit and is not a robust safety proof.
progress=PredictiveStopGate(dict(horizon_s=1.,prediction_step_s=.1,clearance_margin_m=.03,history_bound_scale_m=1.,veto_when_missing=False,uncertainty_in_move=False))
assert not progress.apply(action,missing)[1], 'An unknown peer is not replaced by a fabricated obstacle.'
uncertain=copy.deepcopy(receding);uncertain['formation_history']['neighbor_features'][:,10]=.8
uncertain['local_features'][9]=1.
assert progress.apply(action,uncertain)[1]
uncertain['local_features'][9]=0.;uncertain['local_features'][10]=1.
assert not progress.apply(action,uncertain)[1]
print('Explicit unknown-peer/FORM versus MOVE uncertainty semantics: PASSED')
