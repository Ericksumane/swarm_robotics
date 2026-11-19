"""Physical evaluation of learned acquisition or learned heading alignment."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--task',required=True);p.add_argument('--reward',required=True);p.add_argument('--communication',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--role',choices=['development','final'],required=True);a=p.parse_args()
 if a.episodes<1:p.error('Episodes must be positive.')
 torch.set_num_threads(1);out=Path(a.output);out.mkdir(exist_ok=False)
 task=json.loads(Path(a.task).read_text());comm=json.loads(Path(a.communication).read_text());reward=json.loads(Path(a.reward).read_text());state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 if state['contract']!=task['observation_contract']:raise ValueError('Observation contract mismatch.')
 alignment=task.get('environment')=='alignment';env=(AlignmentEnv if alignment else FormationEnv)(task,comm,reward)
 args,policy,_=make_policy(3,local_dim=state['local_dim'],critic_dim=state['critic_dim'],neighbor_dim=state['neighbor_dim'],geometry_summary=state['geometry_summary'],hold_std_scale=state.get('hold_std_scale'),local_only=state.get('local_only',False));policy.actor.load_state_dict(state['actor']);policy.actor.eval();h,m=hidden_inputs(3,args)
 manifest=dict(task=task,communication=comm,reward=reward,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),seed=a.seed,episodes=a.episodes,role=a.role,component='alignment' if alignment else 'acquisition')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
 for seed in range(a.seed,a.seed+a.episodes):
  obs,_=env.reset(seed);initial=formation_metrics(env.simulation.states,1.,.25);clearance=initial['minimum_clearance_m'];frames=[]
  while True:
   with torch.no_grad():actions,_=policy.act(pack(obs),h,m,deterministic=True)
   obs,_,term,trunc,info=env.step(actions.numpy());geometry=formation_metrics(env.simulation.states,1.,.25);clearance=min(clearance,geometry['minimum_clearance_m'])
   frames.append([[s.x,s.y,s.heading] for s in env.simulation.states])
   if term or trunc:break
  row=dict(seed=seed,reason=info['reason'],elapsed_time_s=info['time_s'],initial_max_pair_error_m=initial['max_pair_error_m'],final_max_pair_error_m=geometry['max_pair_error_m'],minimum_clearance_m=clearance,final_max_speed_mps=max(abs(s.speed) for s in env.simulation.states),final_max_yaw_rate_radps=max(abs(s.yaw_rate) for s in env.simulation.states),final_heading_errors_rad=env.heading_errors().tolist() if alignment else None)
  rows.append(row);(out/f'trajectory_{seed}.json').write_text(json.dumps(frames));print(json.dumps(row),flush=True)
 result=dict(manifest=manifest,results=rows,successes=sum(r['reason']=='success' for r in rows),collisions=sum(r['reason']=='collision' for r in rows))
 (out/'evaluation.json').write_text(json.dumps(result,indent=2));item={k:state[k] for k in ['contract','actor','num_agents','local_dim','critic_dim','neighbor_dim','geometry_summary']};item['local_only']=state.get('local_only',False);item['component']='alignment' if alignment else 'acquisition';item['hold_std_scale']=state.get('hold_std_scale');torch.save(item,out/'actor.pt')
 print(json.dumps({k:result[k] for k in ['successes','collisions']}),flush=True)

if __name__=='__main__':main()
