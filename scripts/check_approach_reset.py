"""Check warm-reset targets, seeded variation and absence of stale packets."""
from dataclasses import asdict
from copy import deepcopy
import json
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.dynamics import RobotState

def main():
 task=json.load(open('configs/tasks/translate_stop_local_goal.json'));comm=json.load(open('configs/communication/nominal.json'))
 reference=MissionEnv(task,comm,{})
 entry=dict(states=[dict(asdict(s),x=s.x+4.4,speed=.08) for s in reference.simulation.states],starts=[p.tolist() for p in reference.starts],goals=[p.tolist() for p in reference.goals])
 task['approach_reset_bank']=dict(entries=[entry],position_jitter_m=.01,heading_jitter_rad=.08,speed_jitter_mps=.02)
 original=deepcopy(task)
 env=MissionEnv(task,comm,{})
 env.reset(900);states=list(env.simulation.states);goals=np.array(env.goals)
 assert all(s.speed>0 for s in states)
 assert all(.5<np.linalg.norm(goals[i]-[s.x,s.y])<.7 for i,s in enumerate(states))
 for i in range(3):
  for j in range(3):
   if i==j:continue
   message,age,valid=env.simulation.network.read(i,j,0.)
   assert valid and age==0.
   assert message.x==states[j].x and message.speed==states[j].speed
 env.reset(900);assert env.simulation.states==states
 env.reset(901);assert env.simulation.states!=states
 np.testing.assert_array_equal(env.goals,goals)
 env.step(np.zeros((3,2)));np.testing.assert_array_equal(env.goals,goals)
 assert task==original
 try:env.simulation.reset_network(900)
 except RuntimeError:pass
 else:raise AssertionError('Mid-episode network reset accepted')
 print('Moving near-goal reset, seeded variation and fixed destinations: PASSED')
 print('Initial receiver cache contains only replaced-state messages: PASSED')

if __name__=='__main__':main()
