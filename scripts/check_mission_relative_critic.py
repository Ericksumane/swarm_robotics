"""Check critic translation invariance and actor information separation."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import critic_input,pack,make_policy,hidden_inputs

def main():
 task=json.load(open('configs/tasks/translate_stop_half_meter_relative.json'))
 env=MissionEnv(task,json.load(open('configs/communication/good.json')),json.load(open('configs/training/translation_reward_goal.json')))
 before=critic_input(env)
 assert before.shape==(3,39)
 raw=pack(env.observe_all())
 shift=np.array([7.,-2.])
 env.simulation.states=[replace(s,x=s.x+shift[0],y=s.y+shift[1]) for s in env.simulation.states]
 env.goals=[g+shift for g in env.goals]
 np.testing.assert_allclose(critic_input(env),before,atol=1e-6)
 # Central critic observes true state; actor cache is independent of that state.
 obs=pack(env.observe_all())
 env.simulation.states[1]=replace(env.simulation.states[1],x=100.)
 assert not np.allclose(critic_input(env),before)
 np.testing.assert_array_equal(pack(env.observe_all()),obs)
 args,policy,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True)
 h,m=hidden_inputs(3,args)
 assert policy.get_values(before,h,m).shape==(3,1)
 print('Relative mission critic translation invariance: PASSED')
 print('Central critic and local actor information separation: PASSED')

if __name__=='__main__':main()
