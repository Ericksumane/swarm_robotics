"""Audit recorded poses and preserve the measured translation research candidate."""
from pathlib import Path
import itertools,json,math,re,shutil
import numpy as np
from swarm_core.mission_env import MissionEnv

def tail_count(values):
 count=0
 for value in values[::-1]:
  if not value:break
  count+=1
 return count

def main():
 root=Path(__file__).resolve().parents[1];folder=root/'experiments/runs/approach_full_validation_v2_001'
 summary=json.loads((folder/'summary.json').read_text());report=json.loads((folder/'composed/evaluation.json').read_text())
 manifest=json.loads((folder/'composed/manifest.json').read_text());task=report['task'];dt=task['simulation']['time_step_s'];mission=task['mission']
 env=MissionEnv(task,report['communication'],manifest['reward']);audits=[];eps=1e-8;needed=math.ceil(mission['success_hold_duration_s']/dt-1e-12)
 for row in report['results']:
  env.reset(row['seed']);initial=np.array([[s.x,s.y,s.heading] for s in env.simulation.states]);goals=np.asarray(env.goals)
  frames=np.asarray(json.loads((folder/'composed'/f"trajectory_{row['seed']}.json").read_text()))
  full=np.concatenate([initial[None,:,:],frames]);turn=(np.diff(full[:,:,2],axis=0)+np.pi)%(2*np.pi)-np.pi
  yaw=turn/dt;speed=np.linalg.norm(np.diff(full[:,:,:2],axis=0),axis=2)/(dt*np.sinc(turn/(2*np.pi)))
  errors=np.linalg.norm(frames[:,:,:2]-goals[None,:,:],axis=2)
  pairs=np.stack([np.linalg.norm(frames[:,i,:2]-frames[:,j,:2],axis=1) for i,j in itertools.combinations(range(3),2)],axis=1)
  pair_error=np.max(np.abs(pairs-task['team']['desired_pair_distance_m']),axis=1)
  point_ok=np.all(errors<=mission['destination_tolerance_m']+eps,axis=1)
  stop_ok=np.all(speed<=mission['stop_linear_speed_tolerance_mps']+eps,axis=1)&np.all(np.abs(yaw)<=mission['stop_angular_speed_tolerance_radps']+eps,axis=1)
  geometry_ok=pair_error<=mission['formation_tolerance_m']+eps
  endpoint=tail_count(point_ok&stop_ok&geometry_ok)>=needed
  assert endpoint==(row['reason']=='success'),f"Pose audit disagrees for seed {row['seed']}"
  point_stop=tail_count(point_ok&stop_ok)>=needed
  failure=None if endpoint else ('goal_position' if tail_count(point_ok)<needed else 'stop_motion' if tail_count(stop_ok)<needed else 'final_formation')
  audits.append(dict(seed=row['seed'],mission_success=endpoint,point_and_stop_success=point_stop,goal_window_ready=tail_count(point_ok)>=needed,stop_window_ready=tail_count(stop_ok)>=needed,geometry_window_ready=tail_count(geometry_ok)>=needed,final_goal_max_m=float(errors[-1].max()),final_pair_error_m=float(pair_error[-1]),final_pair_distances_m=pairs[-1].tolist(),point_and_stop_hold_s=tail_count(point_ok&stop_ok)*dt,final_max_speed_mps=float(speed[-1].max()),final_max_yaw_rate_radps=float(np.abs(yaw[-1]).max()),failure_stage=failure))
 totals=dict(mission_successes=sum(r['mission_success'] for r in audits),point_and_stop_successes=sum(r['point_and_stop_success'] for r in audits),episodes=len(audits),goal_windows_ready=sum(r['goal_window_ready'] for r in audits),stop_windows_ready=sum(r['stop_window_ready'] for r in audits),geometry_windows_ready=sum(r['geometry_window_ready'] for r in audits),failure_stages={key:sum(r['failure_stage']==key for r in audits) for key in ['goal_position','stop_motion','final_formation']})
 (folder/'pose_audit.json').write_text(json.dumps(dict(method='Recorded poses, known fixed goals, exact constant-twist unicycle arc reconstruction; scoring unchanged',totals=totals,results=audits),indent=2))
 candidate=summary['results']['composed'];baseline=summary['results']['cruise']
 assert candidate['successes']>baseline['successes'] and candidate['collisions']<=baseline['collisions']
 assert (folder/'policy_bundle.pt').exists() and (folder/'export_parity.json').exists()
 demo=folder/'gazebo_demo';demo.mkdir(exist_ok=True);success=next(r for r in report['results'] if r['reason']=='success');seed=success['seed'];env.reset(seed)
 demo_manifest=dict(manifest,demo=dict(seed=seed,selected_successful_episode=True,overall_endpoint_successes=candidate['successes'],overall_episodes=len(audits),overall_continuous_formation_successes=candidate['strict_successes']))
 (demo/'manifest.json').write_text(json.dumps(demo_manifest,indent=2));shutil.copy2(folder/'composed'/f'trajectory_{seed}.json',demo/'trajectory.json')
 (demo/'mission_goals.json').write_text(json.dumps(dict(seed=seed,goals=[g.tolist() for g in env.goals]),indent=2))
 frames=json.loads((demo/'trajectory.json').read_text());points=[p[:2] for f in frames for p in f];cx=(min(p[0] for p in points)+max(p[0] for p in points))/2;cy=(min(p[1] for p in points)+max(p[1] for p in points))/2
 xml=(root/'gazebo/replay_gui.config').read_text();xml=re.sub(r'<camera_pose>.*?</camera_pose>',f'<camera_pose>{cx-7} {cy-7} 10 0 0.79 0.785398</camera_pose>',xml);(demo/'gui.config').write_text(xml)
 registry=root/'experiments/selected_translation_candidate.json'
 if registry.exists() and json.loads(registry.read_text()).get('actor_bundle')!=str((folder/'policy_bundle.pt').relative_to(root)):
  backup=root/'experiments/candidate_history/translation_before_approach_v2.json';backup.parent.mkdir(exist_ok=True)
  if not backup.exists():shutil.copy2(registry,backup)
 record=dict(status='research_partial_task_completion_not_hardware_validated',architecture='local_latched_mappo_v2',observation_contract='mission_local_goal_v3',actor_bundle=str((folder/'policy_bundle.pt').relative_to(root)),run=str((folder/'composed').relative_to(root)),replay=str(demo.relative_to(root)),demo_seed=seed,approach_distance_m=.7,actor_goal_horizon_m=.5,requires_local_goal_distance_metadata=True,requires_routing_reset_per_mission=True,evaluation=dict(seed=55000000,episodes=len(audits),baseline=baseline,composed=candidate,pose_audit=totals),scope='Already formed, approximately aligned three-agent synthetic team; good communication; 5 m and 120 s; continuous formation remains unvalidated',full_mission_continuous_formation_validated=False,hardware_validated=False,report='docs/approach_recovery_results.md')
 registry.write_text(json.dumps(record,indent=2))
 rows=report['results'];times=[r['elapsed_time_s'] for r in rows if r['reason']=='success'];compliance=np.mean([r['formation_compliant_fraction'] for r in rows]);worst=max(r['max_pair_error_m'] for r in rows)
 text=f"""# Approach recovery results

## Measured conclusion

The frozen two-actor hierarchy improved full-distance endpoint success from
{baseline['successes']}/{len(rows)} to {candidate['successes']}/{len(rows)} on matched fresh initial states, with
{candidate['collisions']} collisions. Continuous formation success remained
{candidate['strict_successes']}/{len(rows)}. This is a partial research improvement, not completion of all task requirements.

| Fixed 5 m /120 s test | Cruise only | Cruise plus approach |
|---|---:|---:|
| Mission endpoint successes | {baseline['successes']}/{len(rows)} | {candidate['successes']}/{len(rows)} |
| Continuous-formation successes | {baseline['strict_successes']}/{len(rows)} | {candidate['strict_successes']}/{len(rows)} |
| Collisions | {baseline['collisions']} | {candidate['collisions']} |

Fresh final seeds: 55,000,000..55,000,039. The policies were frozen before this
evaluation. Destination tolerance is 0.2 m; final pair tolerance is 0.1 m;
all speeds must remain <=0.03 m/s and yaw rates <=0.05 rad/s for 2 s.
Median successful completion time: {float(np.median(times)):.1f} s.
Mean fraction of steps inside pair tolerance: {100*compliance:.1f}%.
Worst recorded pair error: {worst:.3f} m.

## Failure decomposition

A separate pose audit reconstructed speeds using the exact simulator unicycle
arc and checked the last two seconds. It reproduced every recorded mission
outcome. Point arrival plus stop, excluding geometry, held in
{totals['point_and_stop_successes']}/{len(rows)} cases. Remaining mission failures:
goal position {totals['failure_stages']['goal_position']}; stop motion
{totals['failure_stages']['stop_motion']}; final formation
{totals['failure_stages']['final_formation']}.
Failure categories identify the first unmet condition in position, motion,
then geometry order; they are not mutually independent causal findings.
Independently, goal windows passed in {totals['goal_windows_ready']}/{len(rows)},
stop windows in {totals['stop_windows_ready']}/{len(rows)}, and geometry windows
in {totals['geometry_windows_ready']}/{len(rows)}.
This audit does not change scores or tolerances.

## What changed and what did not

The near-goal policy was trained on moving, laterally displaced simulator
encounters, with no teacher actions. Phase-dependent Gaussian exploration
uses std 0.12 during MOVE and 0.01 during HOLD. It changes exploration,
not learned action means. The 100,096-step checkpoint achieved 11/12 near-goal
development successes; the final 150,000-step checkpoint regressed to 0/12.
Checkpoint selection therefore retained the earlier version.

The first full-distance composition entered approach at 0.5 m and scored 0/40
on 53M seeds. Matching entry to the 0.7 m training encounter distance required
preserving own unclipped goal range for the local selector. Actor inputs and
weights were unchanged. The inspected 53M cases became development evidence;
the corrected version was evaluated on the fresh 55M set above.

Both velocity and yaw commands come from frozen MAPPO actors, shared across
robots. Local task-state routing is a project hierarchy, not a claim of joint
end-to-end training by upstream MAPPO. There is no robot leader or centralized
inference input. The original upstream repository remains unchanged.

## Artifacts and limitations

- Configuration and methods: docs/approach_recovery.md.
- Full results and pose audit: experiments/runs/approach_full_validation_v2_001.
- Actor-only bundle: policy_bundle.pt in that directory; export parity passed.
- Translation registry: experiments/selected_translation_candidate.json.
- Gazebo command: bash scripts/view_translation.sh.
- Gazebo demo seed {seed} is one selected successful episode; aggregate results
  remain {candidate['successes']}/{len(rows)} endpoint and {candidate['strict_successes']}/{len(rows)} continuous formation.

The replay displays Python trajectories on abstract cylinders. It is not a
closed-loop Gazebo physics test. The scene's end pause is display time only.
Initial formation and headings are restricted; poses are ideal in a common
frame. Real hardware interfaces and localization still require confirmation.
The original formation-only candidate is preserved separately.

## Next priority

Reduce residual destination-position errors and improve pair distances during
travel before degraded communication or hardware transfer. Individual 0.2 m goal tolerances alone do
not guarantee 0.1 m pair-distance tolerance; the joint geometry requirement
remains active. Preserve these results as the baseline for that work.
"""
 (root/'docs/approach_recovery_results.md').write_text(text)
 p=root/'README.md';readme=p.read_text();section="""
## Latest translation experiment

Paired 5 m evaluation: 20/40 endpoint successes versus 0/40 for cruise alone,
with no collisions. Continuous-formation success is still 0/40.
This research candidate starts from an existing formation and is not hardware
validated. See docs/approach_recovery_results.md.

View the labelled successful recorded episode with:

```bash
bash scripts/view_translation.sh
```
"""
 if '## Latest translation experiment' not in readme:p.write_text(readme.rstrip()+chr(10)+section)
 print(json.dumps(dict(candidate=record['evaluation'],demo_seed=seed,report='docs/approach_recovery_results.md'),indent=2))

if __name__=='__main__':main()
