"""Check routing uses local masks and preserves all-valid legacy actions."""
import copy,json
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.routed_policy import BatchedRoutedPolicy,RoutedPolicy
from swarm_core.learning.mappo_adapter import pack
torch.set_num_threads(1)
old=torch.load('experiments/runs/geometry_control_001/selected_policy_bundle.pt',weights_only=True)
new=copy.deepcopy(old);new['architecture']='local_network_routed_mappo_v3';new['network_routing']=dict(rule='cruise_degraded_when_any_peer_invalid',required_valid_neighbors=2);new['cruise_degraded']=copy.deepcopy(old['cruise'])
task=json.load(open('configs/tasks/translate_stop_hold_hysteresis.json'));comm=json.load(open('configs/communication/nominal.json'));env=MissionEnv(task,comm,{})
obs,_=env.reset(1);a=BatchedRoutedPolicy(old);b=BatchedRoutedPolicy(new)
x,m=a.act(obs);y,n=b.act(obs);np.testing.assert_array_equal(x,y);assert m==n==['cruise']*3
obs[1]['neighbor_mask'][:]=0;obs[1]['neighbor_features'][:]=0
y,m=b.act(obs);assert m==['cruise','cruise_degraded','cruise']
raw=pack(obs)
for i in range(3):
 individual=RoutedPolicy(new);action,mode=individual.act(raw[i],obs[i]['goal_distance_m'])
 np.testing.assert_allclose(action,y[i],atol=1e-7);assert mode==m[i]
assert b.policy.select_model(raw[1],'approach')=='approach'
print('Local mask-only routing, legacy parity and per-robot isolation: PASSED')
