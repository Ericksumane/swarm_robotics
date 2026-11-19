"""Frozen bundle evaluation with explicit task/observation contracts."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.routed_policy import BatchedRoutedPolicy

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True);parser.add_argument('--task',required=True);parser.add_argument('--communication',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,required=True);parser.add_argument('--episodes',type=int,required=True);parser.add_argument('--role',choices=['development','final'],default='development');cli=parser.parse_args()
 if cli.episodes<1:parser.error('Episodes must be positive.')
 torch.set_num_threads(1);out=Path(cli.output);out.mkdir(exist_ok=False)
 bundle=torch.load(cli.bundle,map_location='cpu',weights_only=True);task=json.loads(Path(cli.task).read_text());comm=json.loads(Path(cli.communication).read_text())
 if bundle['contract']!=task['observation_contract']:raise ValueError('Bundle/task observation contract mismatch.')
 for key in ['hold_entry_tolerance_m','hold_exit_tolerance_m']:
  if key in bundle.get('mission_protocol',{}) and not np.isclose(task['mission'].get(key,task['mission']['destination_tolerance_m']),bundle['mission_protocol'][key]):raise ValueError('Bundle/task mission protocol mismatch.')
 if task.get('approach_reset_bank') or not task.get('start_in_formation') or not np.isclose(task['mission']['translation_distance_m'],5.) or not np.isclose(task['simulation']['episode_time_limit_s'],120.):raise ValueError('Only normal fixed 5 m /120 s mission resets are valid in this benchmark.')
 source=Path(bundle['provenance']['cruise']);manifest=json.loads((source/'manifest.json').read_text())
 env=MissionEnv(task,comm,manifest['reward']);policy=BatchedRoutedPolicy(bundle);rows=[]
 provenance=dict(bundle=str(cli.bundle),bundle_sha256=hashlib.sha256(Path(cli.bundle).read_bytes()).hexdigest(),role=cli.role,seed=cli.seed,episodes=cli.episodes,weights_changed=False)
 (out/'manifest.json').write_text(json.dumps(dict(task=task,communication=comm,reward=manifest['reward'],contract=bundle['contract'],evaluation=provenance),indent=2))
 for seed in range(cli.seed,cli.seed+cli.episodes):
  observations,_=env.reset(seed);policy.reset();start=np.mean([[s.x,s.y] for s in env.simulation.states],axis=0)
  initial_progress=float(np.dot(start-np.mean(env.starts,axis=0),env.direction));assert abs(initial_progress)<1e-9
  errors=[];clearance=float('inf');frames=[]
  while True:
   action,_=policy.act(observations);observations,_,term,trunc,info=env.step(action)
   geometry=formation_metrics(env.simulation.states,1.,.25);errors.append(geometry['max_pair_error_m']);clearance=min(clearance,geometry['minimum_clearance_m']);frames.append([[s.x,s.y,s.heading] for s in env.simulation.states])
   if term or trunc:break
  row=dict(seed=seed,reason=info['reason'],elapsed_time_s=info['time_s'],initial_translation_m=initial_progress,translation_m=info['translation_m'],goal_errors_m=info['goal_errors_m'],max_pair_error_m=max(errors),formation_compliant_fraction=float(np.mean(np.asarray(errors)<=.1)),minimum_clearance_m=clearance,strict_formation_success=bool(info['reason']=='success' and max(errors)<=.1),final_max_speed_mps=max(abs(s.speed) for s in env.simulation.states),final_max_yaw_rate_radps=max(abs(s.yaw_rate) for s in env.simulation.states))
  rows.append(row);(out/f'trajectory_{seed}.json').write_text(json.dumps(frames))
  if seed==cli.seed:(out/'trajectory.json').write_text(json.dumps(frames))
  print(json.dumps(row),flush=True)
 result=dict(provenance=provenance,task=task,communication=comm,results=rows,successes=sum(r['reason']=='success' for r in rows),strict_successes=sum(r['strict_formation_success'] for r in rows),collisions=sum(r['reason']=='collision' for r in rows))
 (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['successes','strict_successes','collisions']}),flush=True)

if __name__=='__main__':main()
