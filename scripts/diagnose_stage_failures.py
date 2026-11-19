"""Reproduce capture geometry and a failing distributed phase transition."""
from pathlib import Path
import json, math
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.staged_policy import BatchedMissionPolicy
root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
entries=json.loads((root/'experiments/runs/arrival_network_recovery_001/training_task.json').read_text())['approach_reset_bank']['entries']
groups={}
for profile in ['good','severe']:
 for label in ['approach_entry','terminal_or_90s']:
  chosen=[e for e in entries if e.get('profile')==profile and e.get('capture')==label]
  errors=np.asarray([[s['x']-g[0],s['y']-g[1]] for e in chosen for s,g in zip(e['states'],e['goals'])])
  headings=[abs(s['heading']) for e in chosen for s in e['states']]
  groups[profile+'_'+label]=dict(mean_longitudinal_error_m=float(errors[:,0].mean()),mean_lateral_error_m=float(errors[:,1].mean()),mean_absolute_lateral_error_m=float(np.abs(errors[:,1]).mean()),max_heading_error_rad=max(headings))
task=json.loads((root/'configs/tasks/dispersed_full_mission.json').read_text());comm=json.loads((root/'configs/communication/nominal.json').read_text())
t=torch.load(root/'experiments/runs/geometry_control_001/selected_policy_bundle.pt',weights_only=True);f=torch.load(root/'experiments/runs/extended_horizon_001/continued_final/actor.pt',weights_only=True)
policy=BatchedMissionPolicy(t,f);env=MissionEnv(task,comm,{})
obs,_=env.reset(63000007);policy.reset();events=[];previous=env.phases.copy()
while True:
 actions,modes=policy.act(obs);obs,_,term,trunc,info=env.step(actions)
 if not np.array_equal(env.phases,previous):
  events.append(dict(time_s=info['time_s'],phases=env.phases.tolist(),headings_rad=[s.heading for s in env.simulation.states],speeds=[s.speed for s in env.simulation.states],max_pair_error_m=formation_metrics(env.simulation.states,1.,.25)['max_pair_error_m']))
  previous=env.phases.copy()
 if term or trunc:break
result=dict(arrival_capture_groups=groups,dispersed_case=dict(seed=63000007,profile='nominal',phase_events=events,reason=info['reason'],elapsed_s=info['time_s']))
dest=root/'experiments/runs/stage_failure_diagnosis_001';dest.mkdir(exist_ok=True);(dest/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
