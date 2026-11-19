"""Check goal clipping changes observations, never physical mission targets."""
from copy import deepcopy
import json
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import pack

def main():
 task=json.load(open('configs/tasks/translate_stop_local_goal.json'))
 comm=json.load(open('configs/communication/good.json'));weights=json.load(open('configs/training/translation_reward_goal.json'))
 env=MissionEnv(task,comm,weights);obs,_=env.reset(818)
 for i,o in enumerate(obs):
  assert np.isclose(np.linalg.norm(o['local_features'][12:14]),.5)
  np.testing.assert_allclose(env.goals[i]-env.starts[i],[5.,0.])
  assert o['neighbor_features'].shape==(2,9)
 # Within the horizon this transform must be identical to the prior contract.
 short=deepcopy(task);short['mission']['translation_distance_m']=.5
 old=deepcopy(short);old['observation_contract']='mission_relational_v2'
 a=MissionEnv(short,comm,weights);b=MissionEnv(old,comm,weights)
 np.testing.assert_allclose(pack(a.reset(818)[0]),pack(b.reset(818)[0]),atol=1e-7)
 goals=np.array(env.goals);env.step(np.zeros((3,2)))
 np.testing.assert_array_equal(env.goals,goals)
 bad=deepcopy(task);bad['goal_observation_horizon_m']=0
 try:MissionEnv(bad,comm,weights)
 except ValueError:pass
 else:raise AssertionError('Invalid horizon accepted')
 print('Bounded local target, unchanged fixed 5 m destination: PASSED')
 print('Short-range input compatibility and invalid horizon rejection: PASSED')

if __name__=='__main__':main()
