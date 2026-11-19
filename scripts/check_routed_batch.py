"""Ensure evaluation batching cannot mix information or mission state between robots."""
import json
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy
from swarm_core.learning.routed_policy import RoutedPolicy,BatchedRoutedPolicy
from swarm_core.mission_env import MissionEnv

def main():
 torch.set_num_threads(1);torch.manual_seed(82)
 bundle=dict(architecture='local_latched_mappo_v2',contract='mission_local_goal_v3',horizon_m=.7,epsilon_m=1e-5,observation_horizon_m=.5)
 for name,scale in [('cruise',None),('approach',.1)]:
  _,p,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True,hold_std_scale=scale)
  bundle[name]=dict(actor=p.actor.state_dict(),num_agents=3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True,hold_std_scale=scale)
 batch=BatchedRoutedPolicy(bundle);individual=[RoutedPolicy(bundle) for _ in range(3)]
 task=json.load(open('configs/tasks/translate_stop_local_goal.json'));env=MissionEnv(task,json.load(open('configs/communication/nominal.json')),{})
 observations=env.reset(81)[0]
 for frame in range(3):
  if frame==1:
   observations[0]['local_features'][12:14]=[.3,0.];observations[0]['goal_distance_m']=.3
  if frame==2:observations=env.reset(81)[0]
  values,modes=batch.act(observations)
  expected=[p.act(o['local_features'].tolist()+o['neighbor_features'].reshape(-1).tolist()+o['neighbor_mask'].tolist(),goal_distance_m=o['goal_distance_m']) for p,o in zip(individual,observations)]
  np.testing.assert_allclose(values,[x[0] for x in expected],atol=1e-6);assert modes==[x[1] for x in expected]
 assert modes==['approach','cruise','cruise']
 batch.reset();assert batch.act(observations)[1]==['cruise']*3
 aligned=json.load(open('configs/tasks/translate_stop_time_aligned.json'));a=MissionEnv(aligned,json.load(open('configs/communication/nominal.json')),{});b=MissionEnv(task,json.load(open('configs/communication/nominal.json')),{})
 for x,y in zip(a.reset(82)[0],b.reset(82)[0]):
  np.testing.assert_array_equal(x['neighbor_features'],y['neighbor_features']);np.testing.assert_array_equal(x['local_features'],y['local_features'])
 print('Batched and separate-robot policy means and latches match: PASSED')
 print('Zero-delay aligned and legacy mission observations are identical: PASSED')

if __name__=='__main__':main()
