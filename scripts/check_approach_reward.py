"""Check that target-bearing reward favors correction, without action labels."""
from dataclasses import replace
import json
import numpy as np
from swarm_core.mission_env import MissionEnv

def main():
 task=json.load(open('configs/tasks/translate_stop_local_goal.json'));comm=json.load(open('configs/communication/nominal.json'))
 weights=json.load(open('configs/training/approach_reward.json'))
 terms=[]
 for sign in [-1,1]:
  env=MissionEnv(task,comm,weights)
  env.simulation.states=[replace(s,heading=.4) for s in env.simulation.states]
  env.simulation._measure();env.simulation.reset_network(71)
  actions=np.tile([0.,sign*.5],(3,1))
  _,_,_,_,info=env.step(actions);terms.append(info['reward_terms'])
 assert terms[0]['goal_heading_progress']>0
 assert terms[1]['goal_heading_progress']<0
 assert terms[0]['goal_heading_absolute']>terms[1]['goal_heading_absolute']
 print('Target-bearing reward favors reducing angular error: PASSED')

if __name__=='__main__':main()
