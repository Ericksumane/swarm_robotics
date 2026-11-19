"""Locate formation violations relative to local approach entry in recorded poses."""
from pathlib import Path
import itertools,json
import numpy as np
from swarm_core.mission_env import MissionEnv

def main():
 root=Path(__file__).resolve().parents[1];out=root/'experiments/runs/control_diagnostics_001';cells={}
 for name in ['legacy_nominal','aligned_nominal','legacy_good','aligned_good']:
  folder=root/'experiments/runs/time_alignment_control_001'/name;report=json.loads((folder/'evaluation.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());env=MissionEnv(report['task'],report['communication'],manifest['reward']);rows=[]
  for row in report['results']:
   env.reset(row['seed']);frames=np.asarray(json.loads((folder/f"trajectory_{row['seed']}.json").read_text()));errors=np.linalg.norm(frames[:,:,:2]-np.asarray(env.goals)[None,:,:],axis=2)
   entered=np.flatnonzero(np.min(errors,axis=1)<.7-1e-5);first=int(entered[0]) if len(entered) else len(frames)-1
   pair=np.max(np.stack([np.abs(np.linalg.norm(frames[:,i,:2]-frames[:,j,:2],axis=1)-1.) for i,j in itertools.combinations(range(3),2)],axis=1),axis=1)
   rows.append(dict(seed=row['seed'],cruise_within_tolerance=bool(pair[:first+1].max()<=.1),cruise_max_pair_error_m=float(pair[:first+1].max()),cruise_compliant_fraction=float(np.mean(pair[:first+1]<=.1)),first_approach_s=(first+1)*.1))
  cells[name]=dict(cruise_successes=sum(r['cruise_within_tolerance'] for r in rows),episodes=len(rows),mean_cruise_compliance=float(np.mean([r['cruise_compliant_fraction'] for r in rows])),results=rows)
 (out/'cruise_segment_audit.json').write_text(json.dumps(cells,indent=2));print(json.dumps({k:{x:v[x] for x in ['cruise_successes','episodes','mean_cruise_compliance']} for k,v in cells.items()},indent=2))

if __name__=='__main__':main()
