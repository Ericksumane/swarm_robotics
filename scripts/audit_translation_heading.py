"""Static heading sensitivity fixtures; not an episode success benchmark."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--output',required=True);cli=parser.parse_args()
 torch.set_num_threads(1);run=Path(cli.run)
 manifest=json.loads((run/'manifest.json').read_text());task=manifest['task']
 task['mission']['translation_distance_m']=5.
 checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
 args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint['local_dim'],critic_dim=checkpoint['critic_dim'],neighbor_dim=checkpoint['neighbor_dim'],geometry_summary=checkpoint['geometry_summary'],hold_std_scale=checkpoint.get('hold_std_scale'))
 policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
 env=MissionEnv(task,json.load(open('configs/communication/nominal.json')),manifest['reward'])
 angles=[-.1,-.05,0.,.05,.1];rows=[]
 for seed in range(43000000,43000020):
  targets=[]
  for angle in angles:
   env.reset(seed)
   env.simulation.states=[replace(s,heading=angle) for s in env.simulation.states]
   env.simulation._measure();env.simulation._transmit();env.simulation.network.deliver(0.)
   h,m=hidden_inputs(3,args)
   with torch.no_grad():action,_=policy.act(pack(env.observe_all()),h,m,deterministic=True)
   targets.append((np.clip(action.numpy()[:,1],-1,1)*task['robot_defaults']['max_yaw_rate_radps']).tolist())
  values=np.asarray(targets)
  rows.append(dict(seed=seed,commanded_yaw_rates_radps=targets,local_slopes_per_s=((values[3]-values[1])/.1).tolist(),zero_heading_bias_radps=values[2].tolist()))
 slopes=np.asarray([r['local_slopes_per_s'] for r in rows]);bias=np.asarray([r['zero_heading_bias_radps'] for r in rows])
 report=dict(scope='Static common-heading perturbation, ideal communication, fixed physical 5 m targets; not success evaluation',run=str(run),angles_rad=angles,mean_local_slope_per_s=float(slopes.mean()),restoring_response_fraction=float(np.mean(slopes<0)),mean_abs_zero_heading_bias_radps=float(np.abs(bias).mean()),rows=rows)
 Path(cli.output).write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__':main()
