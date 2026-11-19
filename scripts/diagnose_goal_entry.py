"""Describe missed destination windows from frozen recorded trajectories."""
from pathlib import Path
import itertools,json
import numpy as np
from swarm_core.mission_env import MissionEnv

def main():
 root=Path(__file__).resolve().parents[1];folder=root/'experiments/runs/approach_full_validation_v2_001'
 report=json.loads((folder/'composed/evaluation.json').read_text());manifest=json.loads((folder/'composed/manifest.json').read_text())
 env=MissionEnv(report['task'],report['communication'],manifest['reward']);rows=[]
 for row in report['results']:
  env.reset(row['seed']);frames=np.asarray(json.loads((folder/'composed'/f"trajectory_{row['seed']}.json").read_text()));goals=np.asarray(env.goals)
  delta=goals[None,:,:]-frames[:,:,:2];distance=np.linalg.norm(delta,axis=2);inside=distance<=.2
  crossings=np.sum(inside[1:]!=inside[:-1],axis=0)
  headings=np.arctan2(np.sin(np.arctan2(delta[:,:,1],delta[:,:,0])-frames[:,:,2]),np.cos(np.arctan2(delta[:,:,1],delta[:,:,0])-frames[:,:,2]))
  per_robot=[]
  for i in range(3):
   times=np.flatnonzero(inside[:,i])
   per_robot.append(dict(robot=i,minimum_error_m=float(distance[:,i].min()),final_error_m=float(distance[-1,i]),entered_goal_region=bool(len(times)),first_entry_s=float((times[0]+1)*.1) if len(times) else None,boundary_crossings=int(crossings[i]),final_goal_bearing_error_rad=float(headings[-1,i]),final_error_xy_m=delta[-1,i].tolist()))
  pair_errors=np.max(np.stack([np.abs(np.linalg.norm(frames[:,i,:2]-frames[:,j,:2],axis=1)-1.) for i,j in itertools.combinations(range(3),2)],axis=1),axis=1)
  violations=np.flatnonzero(pair_errors>.1);first_violation=float((violations[0]+1)*.1) if len(violations) else None
  first_approach=min(t for t in row['first_approach_time_s'] if t is not None)
  rows.append(dict(seed=row['seed'],reason=row['reason'],robots=per_robot,first_formation_violation_s=first_violation,first_approach_s=first_approach,violation_before_approach=bool(first_violation is not None and first_violation<first_approach)))
 failures=[r for r in rows if r['reason']!='success'];never=sum(not p['entered_goal_region'] for r in failures for p in r['robots']);chatter=sum(p['boundary_crossings']>=3 for r in failures for p in r['robots'])
 summary=dict(formation_violations_before_approach=sum(r['violation_before_approach'] for r in rows),episodes=len(rows),failed_episodes=len(failures),robots_never_entering_goal_in_failed_episodes=never,robots_crossing_boundary_at_least_three_times_in_failed_episodes=chatter,scope='Historical diagnostic; these cases are no longer untouched after inspection.')
 out=root/'experiments/runs/control_diagnostics_001';out.mkdir(exist_ok=True);(out/'historical_goal_entry.json').write_text(json.dumps(dict(summary=summary,results=rows),indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
