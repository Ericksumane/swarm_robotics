"""No-pose-edit evaluation of learned acquisition, alignment and translation."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def main():
 p=argparse.ArgumentParser();p.add_argument('--formation',required=True);p.add_argument('--alignment',required=True);p.add_argument('--translation',required=True);p.add_argument('--task',default='configs/tasks/dispersed_full_aligned.json');p.add_argument('--communication',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--time-limit',type=float);a=p.parse_args()
 if a.episodes<1:p.error('Episodes must be positive.')
 torch.set_num_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);task=json.loads(Path(a.task).read_text());comm=json.loads(Path(a.communication).read_text());reward=json.load(open('configs/training/geometry_control.json'))
 if a.time_limit is not None:
  if not np.isfinite(a.time_limit) or a.time_limit<=0:p.error('Time limit must be positive.')
  task['simulation']['episode_time_limit_s']=a.time_limit
 policy=BatchedMissionPolicy(torch.load(a.translation,map_location='cpu',weights_only=True),torch.load(a.formation,map_location='cpu',weights_only=True),alignment=torch.load(a.alignment,map_location='cpu',weights_only=True),acquisition_routing=task['acquisition_routing']);env=MissionEnv(task,comm,reward)
 hashes={name:hashlib.sha256(Path(getattr(a,name)).read_bytes()).hexdigest() for name in ['formation','alignment','translation']}
 manifest=dict(task=task,communication=comm,actors=vars(a),actor_hashes=hashes,seed=a.seed,episodes=a.episodes,role='reserved complete-chain evaluation; not checkpoint selection')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
 for seed in range(a.seed,a.seed+a.episodes):
  obs,_=env.reset(seed);policy.reset();initial=formation_metrics(env.simulation.states,1.,.25);geocount=0;aligncount=0;acquired=None;aligned=None;move=None;post_errors=[];frames=[];clearance=initial['minimum_clearance_m'];mode_counts={};start_headings=None
  while True:
   action,modes=policy.act(obs)
   for mode in modes:mode_counts[mode]=mode_counts.get(mode,0)+1
   obs,_,term,trunc,info=env.step(action);geometry=formation_metrics(env.simulation.states,1.,.25);clearance=min(clearance,geometry['minimum_clearance_m'])
   stopped=all(abs(s.speed)<=.03 and abs(s.yaw_rate)<=.05 for s in env.simulation.states)
   geo=stopped and geometry['max_pair_error_m']<=.1;geocount=geocount+1 if geo else 0
   direction=math.atan2(env.direction[1],env.direction[0]);angles=[abs(math.atan2(math.sin(s.heading-direction),math.cos(s.heading-direction))) for s in env.simulation.states]
   aligncount=aligncount+1 if geo and max(angles)<=.05 else 0
   if geocount>=20 and acquired is None:acquired=info['time_s']
   if aligncount>=20 and aligned is None:aligned=info['time_s']
   if np.all(env.phases>=1):
    if move is None:move=info['time_s'];start_headings=angles
    post_errors.append(geometry['max_pair_error_m'])
   frames.append([[s.x,s.y,s.heading] for s in env.simulation.states])
   if term or trunc:break
  row=dict(seed=seed,reason=info['reason'],elapsed_time_s=info['time_s'],initial_max_pair_error_m=initial['max_pair_error_m'],acquired_time_s=acquired,aligned_time_s=aligned,all_move_time_s=move,move_start_heading_errors_rad=start_headings,goal_errors_m=info['goal_errors_m'],minimum_clearance_m=clearance,mode_counts=mode_counts,strict_post_move_success=bool(info['reason']=='success' and post_errors and max(post_errors)<=.1),max_post_move_pair_error_m=max(post_errors) if post_errors else None,final_max_pair_error_m=geometry['max_pair_error_m'])
  rows.append(row);(out/f'trajectory_{seed}.json').write_text(json.dumps(frames));print(json.dumps(row),flush=True)
 result=dict(manifest=manifest,results=rows,successes=sum(r['reason']=='success' for r in rows),strict_successes=sum(r['strict_post_move_success'] for r in rows),collisions=sum(r['reason']=='collision' for r in rows),acquired=sum(r['acquired_time_s'] is not None for r in rows),aligned=sum(r['aligned_time_s'] is not None for r in rows),started_move=sum(r['all_move_time_s'] is not None for r in rows))
 (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['successes','strict_successes','collisions','acquired','aligned','started_move']}),flush=True)

if __name__=='__main__':main()
