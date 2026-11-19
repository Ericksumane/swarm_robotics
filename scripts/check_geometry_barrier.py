"""Check the soft geometry constraint and zero-weight compatibility."""
import json, math
from dataclasses import replace
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.dynamics import RobotState

task=json.load(open('configs/tasks/translate_stop_local_goal.json'))
comm=json.load(open('configs/communication/nominal.json'))
base=json.load(open('configs/training/geometry_control.json'))
treatment=json.load(open('configs/training/geometry_barrier.json'))
for side,expected in [(1.,0.),(1.09,-.125),(1.2,-.5)]:
 envs=[MissionEnv(task,comm,w) for w in (base,treatment)]
 for env in envs:
  env.reset(123)
  env.simulation.states=[RobotState(x=0.,y=0.),RobotState(x=side,y=0.),RobotState(x=side/2,y=side*math.sqrt(3)/2)]
  env.simulation._measure()
 results=[env.step(np.zeros((3,2))) for env in envs]
 assert np.isclose(results[1][4]['reward_terms']['formation_barrier'],expected)
 assert np.isclose(float(results[1][1][0]-results[0][1][0]),expected,atol=1e-5)
 for a,b in zip(results[0][0],results[1][0]):
  np.testing.assert_array_equal(a['local_features'],b['local_features'])
  np.testing.assert_array_equal(a['neighbor_features'],b['neighbor_features'])
 assert results[0][2:4]==results[1][2:4]
print('Bounded geometry reward and unchanged observations/outcomes: PASSED')
