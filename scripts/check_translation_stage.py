"""Check reset goals, local information boundary, stopping and PPO integration."""
import json
from copy import deepcopy
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv, MOVE
from swarm_core.learning.mappo_adapter import make_policy, pack, critic_input, hidden_inputs

def main():
 task=json.load(open('configs/tasks/translate_stop_aligned.json'))
 comm=json.load(open('configs/communication/good.json'))
 weights=json.load(open('configs/training/translation_reward.json'))
 env=MissionEnv(task,comm,weights)
 obs,_=env.reset(810)
 starts=np.array(env.starts);goals=np.array(env.goals)
 np.testing.assert_allclose(goals-starts,np.tile([5.,0.],(3,1)))
 assert np.all(env.phases==MOVE)
 assert pack(obs).shape==(3,36)
 assert critic_input(env).shape==(3,38)
 assert all(abs(s.heading)<=.05 for s in env.simulation.states)
 # Changing true peer positions without measurements must not leak into actor.
 before=pack(obs).copy()
 from dataclasses import replace
 env.simulation.states[1]=replace(env.simulation.states[1],x=100.)
 np.testing.assert_array_equal(pack(env.observe_all()),before)
 env.reset(810)
 np.testing.assert_allclose(np.array(env.goals),goals)
 blocked=deepcopy(comm);blocked['loss_probability']=1.
 isolated=MissionEnv(task,blocked,weights)
 assert all(np.all(o['neighbor_features']==0) and np.all(o['neighbor_mask']==0) for o in isolated.observe_all())
 # Controlled commands exercise dynamics and stopping; not a learned controller.
 for _ in range(1100):
  actions=np.zeros((3,2))
  for i,s in enumerate(env.simulation.states):
   if env.goals[i][0]-s.x>.15: actions[i,0]=.5
  obs,reward,term,trunc,info=env.step(actions)
  if term or trunc:break
 assert info['reason']=='success',info
 assert all(e<=.2 for e in info['goal_errors_m'])
 assert 4.8<=info['translation_m']<=5.2
 assert all(abs(s.speed)<=.03 for s in env.simulation.states)
 args,policy,trainer=make_policy(3,local_dim=16,critic_dim=38,neighbor_dim=9,geometry_summary=True)
 obs,_=env.reset(811);h,m=hidden_inputs(3,args)
 values,actions,logp,_,_=policy.get_actions(critic_input(env),pack(obs),h,h,m)
 assert values.shape==(3,1) and actions.shape==(3,2)
 assert torch.isfinite(logp).all()
 print('Movement reset, fixed targets and observation boundary: PASSED')
 print('Integrated 5 m displacement and stationary hold: PASSED (test commands)')
 print('Relational mission policy and critic dimensions: PASSED')

if __name__=='__main__':main()
