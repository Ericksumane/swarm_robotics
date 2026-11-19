"""Audit a measured Gazebo closed-loop run without reading a replay input."""
from pathlib import Path
import argparse,json,math

def main():
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);rows=[json.loads(line) for line in (run/'control.jsonl').read_text().splitlines()];manifest=json.loads((run/'manifest.json').read_text());starts=[None]*3;first_success=None;post=[]
 for row in rows:
  policy=row['policy']
  for i,phase in enumerate(policy.get('phases',[])):
   if phase>=1 and starts[i] is None:starts[i]=[row['states'][i]['x'],row['states'][i]['y']]
  if policy.get('phases') and min(policy['phases'])>=1:post.append(policy['max_pair_error_m'])
  if policy.get('mission_completed') and first_success is None:first_success=row
 if first_success is None:first_success=next((r for r in rows if r['policy'].get('reason')=='success'),None)
 direction=manifest.get('hello',{}).get('task',{}).get('mission',{}).get('direction_world',[1.,0.]);norm=math.hypot(*direction);direction=[v/norm for v in direction]
 measured=None
 if first_success is not None:
  measured=[]
  for i,start in enumerate(starts):
   if start is None:raise AssertionError('Success without a measured MOVE start.')
   state=first_success['states'][i];goal=[start[j]+5*direction[j] for j in range(2)];error=math.hypot(state['x']-goal[0],state['y']-goal[1]);assert abs(error-first_success['policy']['goal_errors_m'][i])<1e-7;measured.append(error)
 result=dict(rows=len(rows),simulated_s=rows[-1]['elapsed_s'],max_odometry_age_s=max(max(r['odometry_age_s']) for r in rows),first_success_s=first_success['elapsed_s'] if first_success else None,goal_errors_from_measured_move_start_m=measured,max_post_move_pair_error_m=max(post) if post else None,minimum_clearance_m=min(r['policy'].get('minimum_clearance_m',float('inf')) for r in rows),physics='Gazebo',python_trajectory_input=False)
 (run/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
