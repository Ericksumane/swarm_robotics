"""Paired development probe of two frozen MAPPO actors; no training or action laws."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.approach_router import LocalApproachRouter
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack

def load_actor(folder):
 checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
 args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint['local_dim'],critic_dim=checkpoint['critic_dim'],neighbor_dim=checkpoint['neighbor_dim'],geometry_summary=checkpoint['geometry_summary'],hold_std_scale=checkpoint.get('hold_std_scale'))
 policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
 for p in policy.actor.parameters():p.requires_grad_(False)
 return args,policy

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--communication');parser.add_argument('--cruise-run',default='experiments/runs/translation_heading_precision_001/checkpoint_0120000')
 parser.add_argument('--approach-run',default='experiments/runs/translation_promotion_001/translation_short_relative_001_checkpoint_0030208')
 parser.add_argument('--output',default='experiments/runs/translation_composition_probe_001')
 parser.add_argument('--seed',type=int,default=48000000);parser.add_argument('--episodes',type=int,default=8)
 parser.add_argument('--approach-distance',type=float,default=.5);parser.add_argument('--latch-approach',action='store_true');parser.add_argument('--mode',choices=['both','cruise','composed'],default='both');parser.add_argument('--evaluation-role',choices=['development','final'],default='development');cli=parser.parse_args()
 if cli.episodes<1:parser.error('Episodes must be positive.')
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
 cruise=root/cli.cruise_run;approach=root/cli.approach_run
 args,far=load_actor(cruise);_,near=load_actor(approach)
 out=root/cli.output;out.mkdir(exist_ok=cli.mode!='both')
 manifest=json.loads((cruise/'manifest.json').read_text());task=manifest['task']
 if cli.communication:manifest['communication']=json.loads((root/cli.communication).read_text())
 if task.get('approach_reset_bank') or not task.get('start_in_formation'):
  raise ValueError('Full-distance validation requires a normal zero-progress reset.')
 if not np.isclose(task['mission']['translation_distance_m'],5.) or not np.isclose(task['simulation']['episode_time_limit_s'],120.):
  raise ValueError('This benchmark requires fixed 5 m goals and a 120 s cutoff.')
 if cli.latch_approach:
  near_metadata=torch.load(approach/'checkpoint.pt',map_location='cpu',weights_only=True)
  if task['observation_contract']!='mission_local_goal_v3' or near_metadata['contract']!='mission_local_goal_v3':raise ValueError('Latched bundle requires compatible local-goal contracts.')
  if not np.isclose(task['goal_observation_horizon_m'],.5):raise ValueError('Routing horizon does not match the observation contract.')
 provenance=dict(cruise=str(cruise.relative_to(root)),approach=str(approach.relative_to(root)),horizon_m=.5,approach_distance_m=cli.approach_distance,routing=f'Approach if own unclipped goal range < {cli.approach_distance}-1e-5 or local phase HOLD; retain after entry when latch_approach is true.',source_sha256={str(p.relative_to(root)):hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest() for p in (cruise,approach)},communication_profile=manifest['communication']['profile_name'],weights_changed=False,latch_approach=cli.latch_approach,seed=cli.seed,episodes=cli.episodes,evaluation_role=cli.evaluation_role,scope='Frozen deterministic policy composition; not end-to-end MAPPO retraining or hardware validation')
 summaries={}
 for mode in (['cruise','composed'] if cli.mode=='both' else [cli.mode]):
  folder=out/mode;folder.mkdir();(folder/'manifest.json').write_text(json.dumps(dict(manifest,composition=provenance if mode=='composed' else None),indent=2))
  env=MissionEnv(task,manifest['communication'],manifest['reward']);results=[]
  for seed in range(cli.seed,cli.seed+cli.episodes):
   obs,_=env.reset(seed);initial_progress=float(np.dot(np.mean([[s.x,s.y] for s in env.simulation.states],axis=0)-np.mean(env.starts,axis=0),env.direction));assert abs(initial_progress)<1e-9;h,m=hidden_inputs(3,args);trajectory=[];max_pair_error=0.;within=0;steps=0;clearance=float('inf');switches=[None]*3;routers=[LocalApproachRouter(horizon_m=cli.approach_distance) for _ in range(3)]
   while True:
    raw=pack(obs)
    with torch.no_grad():
     actions,_=far.act(raw,h,m,deterministic=True)
     if mode=='composed':
      near_actions,_=near.act(raw,h,m,deterministic=True)
      use_near=(np.array([o['goal_distance_m'] for o in obs])<cli.approach_distance-1e-5)|(raw[:,11]>.5)
      if cli.latch_approach:
       use_near=np.array([router.select(row,goal_distance_m=o['goal_distance_m'])=='approach' for router,row,o in zip(routers,raw,obs)])
      for i,selected in enumerate(use_near):
       if selected and switches[i] is None:switches[i]=env.simulation.time_s
      actions=torch.where(torch.from_numpy(use_near[:,None]),near_actions,actions)
    obs,_,term,trunc,info=env.step(actions.numpy());steps+=1
    geometry=formation_metrics(env.simulation.states,1.,.25)
    max_pair_error=max(max_pair_error,geometry['max_pair_error_m']);within+=geometry['max_pair_error_m']<=.1;clearance=min(clearance,geometry['minimum_clearance_m'])
    trajectory.append([[s.x,s.y,s.heading] for s in env.simulation.states])
    if term or trunc:break
   row=dict(initial_translation_m=initial_progress,seed=seed,reason=info['reason'],elapsed_time_s=info['time_s'],goal_errors_m=info['goal_errors_m'],translation_m=info['translation_m'],max_pair_error_m=max_pair_error,formation_compliant_fraction=within/steps,minimum_clearance_m=clearance,strict_formation_success=bool(info['reason']=='success' and max_pair_error<=.1),final_max_speed_mps=max(abs(s.speed) for s in env.simulation.states),first_approach_time_s=switches)
   results.append(row);(folder/f'trajectory_{seed}.json').write_text(json.dumps(trajectory))
   if seed==cli.seed:(folder/'trajectory.json').write_text(json.dumps(trajectory))
   print(json.dumps(dict(mode=mode,**row)),flush=True)
  report=dict(task=task,communication=manifest['communication'],provenance=provenance if mode=='composed' else None,results=results,successes=sum(r['reason']=='success' for r in results),strict_successes=sum(r['strict_formation_success'] for r in results),collisions=sum(r['reason']=='collision' for r in results))
  (folder/'evaluation.json').write_text(json.dumps(report,indent=2));summaries[mode]={k:v for k,v in report.items() if k not in ('results','task','communication','provenance')}
 (out/('summary.json' if cli.mode=='both' else 'summary_'+cli.mode+'.json')).write_text(json.dumps(dict(provenance=provenance,results=summaries),indent=2));print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
