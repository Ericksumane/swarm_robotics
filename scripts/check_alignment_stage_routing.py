"""Check learned alignment actions and independent local startup routing."""
import copy,json,math
from dataclasses import replace
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.learning.mappo_adapter import pack
torch.set_num_threads(1)
selection=json.load(open('experiments/runs/dispersed_recovery_validation_001/selection.json'))
translation=torch.load('experiments/runs/local_network_routing_001/selected_policy_bundle.pt',weights_only=True);formation=torch.load(selection['acquisition']['actor'],weights_only=True);alignment=torch.load(selection['alignment']['actor'],weights_only=True)
task=json.load(open('configs/tasks/dispersed_full_aligned.json'));comm=json.load(open('configs/communication/nominal.json'));env=MissionEnv(task,comm,{})
env.reset(1)
# Analytical observation fixture, not a performance rollout or pose correction.
env.simulation.states=[replace(s,x=x,y=y,heading=math.pi/2,speed=0.,yaw_rate=0.) for s,(x,y) in zip(env.simulation.states,[(0.,0.),(1.,0.),(.5,math.sqrt(3)/2)])];env.simulation._measure();env.simulation.reset_network(1);obs=env.observe_all()
policy=BatchedMissionPolicy(translation,formation,alignment=alignment,acquisition_routing=task['acquisition_routing'])
for _ in range(19):_,modes=policy.act(obs);assert modes==['formation']*3
actions,modes=policy.act(obs);assert modes==['alignment']*3
model,h,m=policy.alignment_model
for i,row in enumerate(pack(obs)):
 with torch.no_grad():expected,_=model.act(row[None,:],h,m,deterministic=True)
 np.testing.assert_allclose(actions[i],expected.numpy()[0],atol=1e-7)
mixed=copy.deepcopy(obs);mixed[1]['neighbor_mask'][:]=0;mixed[1]['neighbor_features'][:]=0
_,modes=policy.act(mixed);assert modes==['alignment','formation','alignment']
policy.reset();assert not any(r.alignment_latched for r in policy.acquisition_routers)
print('Learned alignment parity, local state isolation and reset: PASSED')
