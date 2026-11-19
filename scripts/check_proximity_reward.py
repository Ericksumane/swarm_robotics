"""Check the pre-contact reward without adding a motion controller."""
import json,math
import numpy as np
from swarm_core.dynamics import RobotState
from swarm_core.formation_env import FormationEnv
def main():
 task=json.load(open('configs/tasks/formation_v1.json'));comm=json.load(open('configs/communication/nominal.json'));weights=json.load(open('configs/training/formation_reward_proximity.json'))
 env=FormationEnv(task,comm,weights)
 env.reset();env.simulation.states=[RobotState(x=0,y=0),RobotState(x=1,y=0),RobotState(x=.5,y=math.sqrt(3)/2)]
 _,_,_,_,info=env.step(np.zeros((3,2)))
 assert info['reward_terms']['proximity']==0
 env.reset();env.simulation.states=[RobotState(x=0,y=0),RobotState(x=.6,y=0),RobotState(x=0,y=2)]
 _,_,term,_,info=env.step(np.zeros((3,2)))
 assert not term and info['reward_terms']['proximity']<0
 print('Pre-contact proximity penalty and unpenalized target formation: PASSED')
if __name__=='__main__':main()
