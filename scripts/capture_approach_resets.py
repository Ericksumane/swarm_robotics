"""Capture simulator encounter states as reset curricula; no expert actions."""
from pathlib import Path
from dataclasses import asdict
import json
import torch
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs

def main():
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
 source=root/'experiments/runs/translation_heading_precision_001/checkpoint_0120000'
 manifest=json.loads((source/'manifest.json').read_text());checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
 args,policy,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True)
 policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
 out=root/'experiments/runs/approach_capture_001';out.mkdir(exist_ok=False)
 for name,first,count in [('training',49000000,40),('heldout',50000000,20)]:
  entries=[]
  for seed in range(first,first+count):
   env=MissionEnv(manifest['task'],manifest['communication'],manifest['reward']);obs,_=env.reset(seed);h,m=hidden_inputs(3,args)
   while True:
    with torch.no_grad():actions,_=policy.act(pack(obs),h,m,deterministic=True)
    obs,_,term,trunc,info=env.step(actions.numpy())
    if min(info['goal_errors_m'])<=.7 and max(info['goal_errors_m'])<=1.2:
     entries.append(dict(source_seed=seed,source_time_s=info['time_s'],states=[asdict(s) for s in env.simulation.states],starts=[p.tolist() for p in env.starts],goals=[p.tolist() for p in env.goals]));break
    if term or trunc:raise RuntimeError(f'No encounter captured: {seed}')
   print(json.dumps(dict(pool=name,captured=len(entries),seed=seed)),flush=True)
  task=json.loads(json.dumps(manifest['task']));task['task_name']='approach_encounter_'+name+'_v1';task['simulation']['episode_time_limit_s']=45.
  task['approach_reset_bank']=dict(entries=entries,position_jitter_m=.01,heading_jitter_rad=.08,speed_jitter_mps=.02)
  (out/(name+'_task.json')).write_text(json.dumps(task,indent=2))
 (out/'provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(root)),training_source_seeds=[49000000,49000039],heldout_source_seeds=[50000000,50000019],scope='Synthetic reset states near destination; no actions, teacher labels or real robot data'),indent=2))
 print('Approach reset pools captured.',flush=True)

if __name__=='__main__':main()
