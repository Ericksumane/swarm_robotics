"""Check local phase selection and exact formation-actor projection."""
import json
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.learning.mappo_adapter import pack
torch.set_num_threads(1)
t=torch.load('experiments/runs/geometry_control_001/selected_policy_bundle.pt',weights_only=True)
f=torch.load('experiments/runs/extended_horizon_001/continued_final/actor.pt',weights_only=True)
e=MissionEnv(json.load(open('configs/tasks/dispersed_full_mission.json')),json.load(open('configs/communication/nominal.json')), {})
obs,_=e.reset(63000000);model=BatchedMissionPolicy(t,f);actions,modes=model.act(obs)
assert modes==['formation']*3 and not any(r.approach_latched for r in model.routers)
raw=pack(obs);projected=np.concatenate([raw[:,:9],raw[:,16:]],axis=1)
with torch.no_grad():expected,_=model.formation.act(projected,np.repeat(model.form_hidden,3,axis=0),np.repeat(model.form_masks,3,axis=0),deterministic=True)
np.testing.assert_allclose(actions,expected.numpy(),atol=1e-7)
e.phases[:]=1;e.starts=[np.array([s.x,s.y]) for s in e.simulation.estimates];e.goals=[p+[5.,0.] for p in e.starts]
actions,modes=model.act(e.observe_all());assert modes==['cruise']*3 and np.isfinite(actions).all()
model.reset();assert not any(r.approach_latched for r in model.routers)
print('Local staged routing, projection parity and reset: PASSED')
