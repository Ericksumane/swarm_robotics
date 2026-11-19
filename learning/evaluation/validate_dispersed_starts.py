"""Separate acquisition and complete-mission validation from dispersed starts."""
from pathlib import Path
import concurrent.futures,json,hashlib,copy
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.mission_env import MissionEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def main():
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/dispersed_validation_001';out.mkdir(exist_ok=True)
 actor_path=root/'experiments/runs/extended_horizon_001/continued_final/actor.pt';formation=torch.load(actor_path,map_location='cpu',weights_only=True)
 translation_path=root/'experiments/runs/geometry_control_001/selected_policy_bundle.pt';translation=torch.load(translation_path,map_location='cpu',weights_only=True)
 source=json.loads((root/'experiments/runs/extended_horizon_001/continued_final/manifest.json').read_text())
 cells=[(mode,profile) for mode in ['formation','full_mission'] for profile in ['nominal','good']]
 def run(cell):
  mode,profile=cell;dest=out/(mode+'_'+profile)
  if (dest/'evaluation.json').exists():
   result=json.loads((dest/'evaluation.json').read_text());return mode+'_'+profile,{k:result[k] for k in ['formation_acquired','full_translation_successes','strict_post_move_successes','collisions','initially_formed']}
  dest.mkdir(exist_ok=True);task=json.loads((root/('configs/tasks/formation_relational_broad_evaluation.json' if mode=='formation' else 'configs/tasks/dispersed_full_mission.json')).read_text());comm=json.loads((root/f'configs/communication/{profile}.json').read_text())
  count=20 if mode=='formation' else 10
  reward=source['reward'] if mode=='formation' else json.loads((root/'configs/training/geometry_control.json').read_text())
  env=(FormationEnv if mode=='formation' else MissionEnv)(task,comm,reward)
  if mode=='formation':
   args,policy,_=make_policy(3,local_dim=9,critic_dim=formation['critic_dim'],neighbor_dim=9,geometry_summary=True);policy.actor.load_state_dict(formation['actor']);policy.actor.eval();h,m=hidden_inputs(3,args)
  else:policy=BatchedMissionPolicy(translation,formation)
  manifest=dict(task=task,communication=comm,reward=reward,mode=mode,episodes=count,seed=63000000,role='diagnostic validation, no training on these starts',formation_actor_sha256=hashlib.sha256(actor_path.read_bytes()).hexdigest(),translation_bundle_sha256=hashlib.sha256(translation_path.read_bytes()).hexdigest())
  (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
  for seed in range(63000000,63000000+count):
   obs,_=env.reset(seed)
   if mode=='full_mission':policy.reset()
   initial=formation_metrics(env.simulation.states,1.,.25);hold=0;acquired=None;all_move=None;post_move_errors=[];frames=[];clearance=initial['minimum_clearance_m']
   while True:
    if mode=='formation':
     with torch.no_grad():actions,_=policy.act(pack(obs),h,m,deterministic=True)
     actions=actions.numpy()
    else:actions,_=policy.act(obs)
    obs,_,term,trunc,info=env.step(actions);geometry=formation_metrics(env.simulation.states,1.,.25);clearance=min(clearance,geometry['minimum_clearance_m'])
    stopped=all(abs(s.speed)<=.03 and abs(s.yaw_rate)<=.05 for s in env.simulation.states)
    hold=hold+1 if stopped and geometry['max_pair_error_m']<=.1 else 0
    if hold>=20 and acquired is None:acquired=info['time_s']
    if mode=='full_mission' and np.all(env.phases>=1):
     if all_move is None:all_move=info['time_s']
     post_move_errors.append(geometry['max_pair_error_m'])
    if seed==63000000:frames.append([[s.x,s.y,s.heading] for s in env.simulation.states])
    if term or trunc:break
   row=dict(seed=seed,reason=info['reason'],initial_max_pair_error_m=initial['max_pair_error_m'],formation_acquired=acquired is not None,acquisition_time_s=acquired,all_agents_started_move_time_s=all_move,elapsed_time_s=info['time_s'],minimum_clearance_m=clearance,final_max_pair_error_m=geometry['max_pair_error_m'],goal_errors_m=info.get('goal_errors_m'),full_translation_success=bool(mode=='full_mission' and info['reason']=='success'),strict_post_move_success=bool(mode=='full_mission' and info['reason']=='success' and post_move_errors and max(post_move_errors)<=.1))
   rows.append(row);print(mode,profile,json.dumps(row),flush=True)
   if seed==63000000:(dest/'trajectory.json').write_text(json.dumps(frames))
  result=dict(manifest=manifest,results=rows,formation_acquired=sum(r['formation_acquired'] for r in rows),full_translation_successes=sum(r['full_translation_success'] for r in rows),strict_post_move_successes=sum(r['strict_post_move_success'] for r in rows),collisions=sum(r['reason']=='collision' for r in rows),initially_formed=sum(r['initial_max_pair_error_m']<=.1 for r in rows))
  (dest/'evaluation.json').write_text(json.dumps(result,indent=2));return mode+'_'+profile,{k:result[k] for k in ['formation_acquired','full_translation_successes','strict_post_move_successes','collisions','initially_formed']}
 # Model construction consumes torch RNG; sequential cells avoid concurrent initialization.
 results=dict(run(cell) for cell in cells)
 (out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
