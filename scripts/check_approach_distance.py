"""Check that local routing can precede actor feature saturation."""
import json
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import pack
from swarm_core.learning.approach_router import LocalApproachRouter

def main():
 task=json.load(open('configs/tasks/translate_stop_local_goal.json'));env=MissionEnv(task,json.load(open('configs/communication/nominal.json')),{})
 observation=env.reset(818)[0][0];raw=pack([observation])[0]
 assert np.isclose(observation['goal_distance_m'],5.) and np.isclose(np.linalg.norm(raw[12:14]),.5)
 router=LocalApproachRouter(horizon_m=.7)
 assert router.select(raw,goal_distance_m=5.)=='cruise'
 try:router.select(raw)
 except ValueError:pass
 else:raise AssertionError('Missing full range accepted')
 assert router.select(raw,goal_distance_m=.6)=='approach'
 assert router.select(raw,goal_distance_m=5.)=='approach'
 router.reset();assert router.select(raw,goal_distance_m=5.)=='cruise'
 print('Unclipped local range drives entry without changing actor features: PASSED')
 print('Range metadata, local latch and reset contracts: PASSED')

if __name__=='__main__':main()
