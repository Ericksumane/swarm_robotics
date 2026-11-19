"""Reproduce old collision starts and log physical closest-approach warning."""
from pathlib import Path
import json,math
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack
from swarm_core.geometry import formation_metrics
from swarm_core.collision_risk import closing_risk

torch.set_num_threads(1)
task=json.load(open('configs/tasks/dispersed_history_evaluation.json'));comm=json.load(open('configs/communication/nominal.json'));reward=json.load(open('configs/training/dispersed_acquisition_reward.json'))
item=torch.load('experiments/runs/bounded_history_001/export_50k/actor.pt',map_location='cpu',weights_only=True)
args,policy,_=make_policy(3,local_dim=9,neighbor_dim=11,critic_dim=item['critic_dim'],geometry_summary=True);policy.actor.load_state_dict(item['actor']);policy.actor.eval();h,m=hidden_inputs(3,args)
rows=[]
for seed in (83000003,83000005):
    env=FormationEnv(task,comm,reward);obs,_=env.reset(seed);trace=[]
    while True:
        with torch.no_grad():actions,_=policy.act(pack(obs),h,m,deterministic=True)
        obs,_,term,trunc,info=env.step(actions.numpy());geometry=formation_metrics(env.simulation.states,1.,.25)
        trace.append(dict(time_s=info['time_s'],clearance_m=geometry['minimum_clearance_m'],closing_risk=closing_risk(env.simulation.states,.25,2.,.3),states=[vars(s) for s in env.simulation.states],actions=actions.tolist()))
        if term or trunc:break
    warnings=[r for r in trace if r['closing_risk']>1.]
    row=dict(seed=seed,reason=info['reason'],time_s=info['time_s'],first_predicted_overlap_s=warnings[0]['time_s'] if warnings else None,warning_lead_s=info['time_s']-warnings[0]['time_s'] if warnings else None,trace=trace)
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
Path('experiments/runs/history_coordination_001/collision_diagnosis.json').write_text(json.dumps(rows,indent=2))
