"""Capture approach resets, train only the near actor, gate regressions."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,subprocess,sys
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.routed_policy import BatchedRoutedPolicy

def main():
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/arrival_network_recovery_001';out.mkdir(exist_ok=False)
 source_path=root/'experiments/runs/geometry_control_001/selected_policy_bundle.pt';source=torch.load(source_path,map_location='cpu',weights_only=True)
 task=json.loads((root/'configs/tasks/translate_stop_hold_hysteresis.json').read_text());weights=json.loads((root/'configs/training/approach_reward.json').read_text());entries=[]
 for profile in ['good','severe']:
  comm=json.loads((root/f'configs/communication/{profile}.json').read_text());env=MissionEnv(task,comm,weights);policy=BatchedRoutedPolicy(source)
  for seed in range(64000000,64000006):
   obs,_=env.reset(seed);policy.reset();captured=False
   def capture(label):
    entries.append(dict(seed=seed,profile=profile,capture=label,states=[dict(x=s.x,y=s.y,heading=s.heading,speed=s.speed,yaw_rate=s.yaw_rate) for s in env.simulation.states],starts=[p.tolist() for p in env.starts],goals=[p.tolist() for p in env.goals]))
   while True:
    if not captured and min(o['goal_distance_m'] for o in obs)<.7:
     capture('approach_entry');captured=True
    actions,_=policy.act(obs);obs,_,term,trunc,info=env.step(actions)
    if term or trunc or env.simulation.time_s>=90.:
     capture('terminal_or_90s');break
 old=json.loads((root/'experiments/runs/approach_recovery_phase_noise_001/training_task.json').read_text())
 training=copy.deepcopy(task);training['task_name']='arrival_network_recovery_training_v1';training['simulation']['episode_time_limit_s']=45.
 training['approach_reset_bank']=dict(entries=entries+old['approach_reset_bank']['entries'][:12],position_jitter_m=.01,heading_jitter_rad=.08,speed_jitter_mps=.02)
 training['exploration']=old['exploration'];training_path=out/'training_task.json';training_path.write_text(json.dumps(training,indent=2))
 resume=root/'experiments/runs/approach_recovery_phase_noise_001/checkpoint_0100096/checkpoint.pt'
 with (out/'training.log').open('w') as log:subprocess.run([sys.executable,'scripts/train_formation.py','--steps','40000','--snapshot-interval','10000','--seed','452','--gamma','.99','--actor-lr','.00003','--initial-std','.04','--final-std','.04','--entropy-coef','0','--resume',str(resume),'--task',str(training_path),'--reward','configs/training/approach_reward.json','--communication-mixture','configs/training/arrival_communication_mixture.json','--output',str(out/'training')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 def evaluate(bundle,dest,profile,seed,episodes,role):
  with dest.with_suffix('.log').open('w') as log:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 candidates=[];development={}
 for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
  state=torch.load(checkpoint,map_location='cpu',weights_only=True);b=copy.deepcopy(source);b['approach']={k:state[k] for k in b['approach']};b['status']='unvalidated_severe_arrival_candidate';b['adaptation']=dict(type='approach-only network-mixture MAPPO continuation',source_bundle_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),checkpoint=str(checkpoint),cruise_weights_changed=False)
  path=out/(checkpoint.stem+'_bundle.pt');torch.save(b,path)
  def dev(profile):return profile,evaluate(path,out/(checkpoint.stem+'_'+profile+'_dev'),profile,65000000,4,'development')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(dev,['nominal','good','severe']))
  development[checkpoint.stem]={p:{k:d[k] for k in ['successes','strict_successes','collisions']} for p,d in results.items()}
  if all(results[p]['strict_successes']==4 and results[p]['collisions']==0 for p in ['nominal','good']):
   severe=results['severe'];score=(severe['strict_successes'],severe['successes'],-sum(max(r['goal_errors_m']) for r in severe['results']))
   candidates.append((score,path))
 selected=max(candidates,key=lambda x:x[0])[1] if candidates else source_path
 selection=dict(selected_bundle=str(selected),development=development,selection_role='Development only; nominal/good strict gates then severe performance; frozen before reserved tests')
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 def final(profile):
  d=evaluate(selected,out/(profile+'_final'),profile,66000000,40,'final');return profile,{k:d[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,['nominal','good','moderate','severe']))
 (out/'summary.json').write_text(json.dumps(dict(selection=selection,results=results,episodes=40,seed=66000000),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
