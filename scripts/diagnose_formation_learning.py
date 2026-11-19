"""Controlled reward ranking and upstream PPO gradient-direction diagnosis.

Uses synthetic state fixtures, not expert demonstrations or a deployable policy.
The temporary bandit-like PPO policy is discarded after the diagnostic.
"""
import json,math
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from swarm_core.dynamics import RobotState
from swarm_core.formation_env import FormationEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,pack,critic_input,hidden_inputs

def load(path):return json.loads(Path(path).read_text())

def main():
 torch.set_num_threads(1);torch.manual_seed(91)
 task=load('configs/tasks/formation_stage1_geometry.json')
 comm=load('configs/communication/nominal.json');weights=load('configs/training/formation_reward_proximity.json')
 positions=np.array([[0.,0.],[1.2,0.],[.6,1.2*math.sqrt(3)/2]])
 center=positions.mean(axis=0)
 headings=[math.atan2(center[1]-y,center[0]-x) for x,y in positions]
 initial=[RobotState(float(x),float(y),heading) for (x,y),heading in zip(positions,headings)]
 def fixture(away=False):
  env=FormationEnv(task,comm,weights);env.reset(seed=91)
  env.simulation.states=[replace(s,heading=(s.heading+math.pi if away else s.heading)) for s in initial]
  env.simulation._measure();env.simulation._transmit();env.simulation.network.deliver(0.)
  return env
 cases={}
 for name,linear in [('stationary',0.),('inward',.4),('outward',-.4)]:
  env=fixture();total=0.;trajectory=[]
  for t in range(30):
   action=np.zeros((3,2));action[:,0]=linear if t<9 else 0.
   _,reward,term,trunc,info=env.step(action)
   total+=float(reward[0]);trajectory.append(float(reward[0]))
   if term or trunc:break
  geometry=formation_metrics(env.simulation.states,1.,.25)
  cases[name]=dict(return_sum=total,final_rmse_m=geometry['formation_rmse_m'],reason=info['reason'],rewards=trajectory)
 assert cases['inward']['return_sum']>cases['stationary']['return_sum']>cases['outward']['return_sum']
 assert cases['inward']['final_rmse_m']<cases['stationary']['final_rmse_m']
 print('Controlled reward ranking:',json.dumps({k:{a:b for a,b in v.items() if a!='rewards'} for k,v in cases.items()}),flush=True)

 args,policy,trainer=make_policy(3,neighbor_dim=9)
 with torch.no_grad():policy.actor.act.action_out.logstd._bias.fill_(math.log(.3))
 base=[fixture(False),fixture(True)]
 raw=np.concatenate([np.tile(pack(env.simulation.observe_all()),(256,1)) for env in base])
 central=np.concatenate([np.tile(critic_input(env),(256,1)) for env in base])
 h,m=hidden_inputs(len(raw),args)
 with torch.no_grad():values,actions,logps,_,_=policy.get_actions(central,raw,h,h,m)
 def deterministic_rewards():
  results=[]
  for away in [False,True]:
   env=fixture(away);hh,mm=hidden_inputs(3,args)
   with torch.no_grad():action,_=policy.act(pack(env.simulation.observe_all()),hh,mm,deterministic=True)
   _,r,_,_,_=env.step(action.numpy())
   results.append(dict(away=away,mean_linear_action=float(action[:,0].mean()),reward=float(r[0])))
  return results
 before=deterministic_rewards();sampled=actions.numpy().reshape(2,256,3,2);rewards=[]
 for group in range(2):
  for action in sampled[group]:
   env=fixture(bool(group));_,r,_,_,_=env.step(action);rewards.extend(r)
 reward=np.asarray(rewards,np.float32)[:,None]
 advantage=(reward-reward.mean())/(reward.std()+1e-5)
 sample=(central,raw,h,h.copy(),actions.numpy(),values.numpy(),reward,m,m.copy(),logps.numpy(),advantage,None)
 for _ in range(12):trainer.ppo_update(sample)
 after=deterministic_rewards()
 # Fresh on-policy batches with a state-conditioned Monte Carlo baseline.
 # Each initial state has repeated independent samples; no action labels.
 fresh_history=[]
 for batch in range(20):
  with torch.no_grad():values,actions,logps,_,_=policy.get_actions(central,raw,h,h,m)
  sampled=actions.numpy().reshape(2,256,3,2);all_rewards=[]
  for group in range(2):
   group_rewards=[]
   for action in sampled[group]:
    env=fixture(bool(group));_,r,_,_,_=env.step(action);group_rewards.append(float(r[0]))
   all_rewards.append(group_rewards)
  group_rewards=np.asarray(all_rewards,np.float32)
  # Leave-one-out baseline uses other independent joint-action samples.
  baseline=(group_rewards.sum(axis=1,keepdims=True)-group_rewards)/(group_rewards.shape[1]-1)
  advantage=np.repeat((group_rewards-baseline).reshape(-1),3)[:,None]
  advantage=(advantage-advantage.mean())/(advantage.std()+1e-5)
  reward=np.repeat(group_rewards.reshape(-1),3)[:,None]
  sample=(central,raw,h,h.copy(),actions.numpy(),values.numpy(),reward,m,m.copy(),logps.numpy(),advantage,None)
  for _ in range(4):trainer.ppo_update(sample)
  if batch in (0,4,9,19):
   measured=deterministic_rewards();fresh_history.append(dict(batch=batch+1,measurements=measured))
   print('Fresh-batch diagnostic:',json.dumps(fresh_history[-1]),flush=True)
 fresh_after=deterministic_rewards()
 result=dict(fresh_batch_history=fresh_history,fresh_batch_after=fresh_after,reward_ranking=cases,ppo_direction_before=before,ppo_direction_after=after,
  controlled_ppo_improved=all(a['reward']>b['reward'] for a,b in zip(after,before)))
 output=Path('experiments/runs/reward_gradient_diagnostic.json');output.write_text(json.dumps(result,indent=2))
 print('PPO before:',json.dumps(before),flush=True)
 print('PPO after:',json.dumps(after),flush=True)
 print('Controlled PPO improved:',result['controlled_ppo_improved'])
 print('No diagnostic weights were exported or used as a robot controller.')

if __name__=='__main__':main()
