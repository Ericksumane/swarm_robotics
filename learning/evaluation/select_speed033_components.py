"""Select a documented component adaptation only after paired/new-start checks."""
from pathlib import Path
import hashlib,json,time,math
import torch
from swarm_core.mission_spec import completion_task
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/runs/speed033_campaign_001'

def load(path):return json.loads(path.read_text())
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def ready(path,count):
 try:
  d=load(path);return d if set(d['profiles'])=={'nominal','good','moderate','severe'} and all(len(v)==count for v in d['profiles'].values()) else None
 except (FileNotFoundError,json.JSONDecodeError):return None

def main():
 requested={'evaluation_baseline':4,'evaluation_formation_adapted':4,'confirmation_baseline':2,'confirmation_formation_adapted':2};deadline=time.monotonic()+900
 while True:
  reports={name:ready(BASE/name/'evaluation.json',count) for name,count in requested.items()}
  gzpaths=[BASE/(name+'/summary.json') for name in ['gazebo_nominal_adapted','gazebo_good_adapted','gazebo_hard_severe_adapted']]
  if all(reports.values()) and all(p.exists() for p in gzpaths):break
  if time.monotonic()>deadline:raise RuntimeError('Evaluation still incomplete; no selection performed.')
  time.sleep(10)
 candidate=torch.load(BASE/'formation_adapted_bundle.pt',map_location='cpu',weights_only=True);comparisons={};audit=[];safe=True
 for group in ['evaluation','confirmation']:
  base=reports[group+'_baseline'];adapted=reports[group+'_formation_adapted'];scores={}
  for profile in base['profiles']:
   br=base['profiles'][profile];ar=adapted['profiles'][profile];assert [r['seed'] for r in br]==[r['seed'] for r in ar]
   bs=sum(r['reason']=='success' for r in br);ss=sum(r['reason']=='success' for r in ar);collisions=sum(r['reason']=='collision' for r in ar);minimum=min(r['minimum_clearance_m'] for r in ar)
   safe=safe and ss>=bs and collisions==0 and minimum>=.05
   tolerance=completion_task(candidate['task'],profile)['mission']['formation_tolerance_m']
   for row in ar:
    frames=load(BASE/(group+'_formation_adapted')/(profile+'_'+str(row['seed'])+'.json'));held=[f for f in frames if row['first_success_s'] is not None and f['time_s']>=row['first_success_s']-1e-8]
    violations=sum(any(e is None or e>.2 for e in f['goal_errors_m']) or f['pair_error_m']>tolerance or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']) for f in held)
    if row['reason']=='success':safe=safe and len(held)>=600 and violations==0
    audit.append(dict(group=group,profile=profile,seed=row['seed'],success=row['reason']=='success',held_samples=len(held),hold_violations=violations))
   scores[profile]=dict(baseline_successes=bs,adapted_successes=ss,runs=len(ar),adapted_collisions=collisions,adapted_minimum_clearance_m=minimum)
  comparisons[group]=scores
 physical={}
 for path in gzpaths:
  d=load(path);r=d['final_policy'];physical[path.parent.name]=dict(completed=r['mission_completed'],first_success_s=r['first_success_s'],goal_errors_m=r['goal_errors_m'],reason=d['reason']);safe=safe and r['mission_completed']
 hard=BASE/'gazebo_hard_severe_adapted';d=load(hard/'summary.json');start=d['final_policy']['first_success_s'];violations=0;samples=0;minimum=math.inf
 for line in (hard/'control.jsonl').open():
  f=json.loads(line);r=f['policy'];minimum=min(minimum,r['minimum_clearance_m'])
  if start is not None and r['time_s']>=start-1e-8:
   samples+=1;violations+=int(any(e is None or e>.2 for e in r['goal_errors_m']) or r['max_pair_error_m']>.2 or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']))
 safe=safe and minimum>=.05 and violations==0 and samples>=600
 # This gate is finite regression evidence, never final or physical approval.
 record=load(BASE/'formation_adapted_candidate.json');baseline_hash=hashlib.sha256((BASE/'baseline_bundle.pt').read_bytes()).hexdigest();current=load(ROOT/'experiments/selected_delivery_candidate.json');promoted=bool(safe and current['sha256']==baseline_hash)
 record.update(status='speed033_component_selected_pending_final_acceptance' if promoted else 'component_candidate_not_promoted',report='docs/speed033_component_selection.md',trained_experts=['formation_history','alignment'],retained_baseline_experts=['formation_fresh','cruise','cruise_degraded','approach'],campaign_steps_per_profile=200000,selected_expert_training_steps_per_profile=dict(formation_history=90000,alignment=10000),hardware_approved=False)
 write(BASE/'formation_adapted_candidate.json',record)
 if promoted:
  write(ROOT/'experiments/selected_delivery_candidate.json',record);p=ROOT/'configs/deployment/delivery_candidate.json';m=load(p);m.update(actor_bundle=record['actor_bundle'],actor_bundle_sha256=record['sha256'],release_id='speed033_selected_001',status=record['status'],policy_final_acceptance_met=False);write(p,m)
 summary=dict(status='complete',promoted=promoted,comparisons=comparisons,physical=physical,hold_audit=audit,hard_severe_hold=dict(samples=samples,violations=violations,minimum_clearance_m=minimum),candidate=record,scope='Development component selection plus two new paired starts/profile; small finite samples, not final acceptance or hardware approval. No additional training beyond 200000 steps/profile.')
 write(BASE/'component_selection_summary.json',summary)
 table=[]
 for group,scores in comparisons.items():
  for profile,s in scores.items():table.append('| '+group+' | '+profile+' | '+str(s['baseline_successes'])+'/'+str(s['runs'])+' | '+str(s['adapted_successes'])+'/'+str(s['runs'])+' | '+format(s['adapted_minimum_clearance_m'],'.3f')+' |')
 text='''# Speed 0.33 component selection

The requested campaign completed 200,000 environment steps per communication
profile (800,000 total) across all six experts. The fully adapted model was
retained but not promoted because nominal/good completion regressed. See
speed033_training_results.md for that negative result; it must not be hidden.

A subsequent DEVELOPMENT selection retains trained history-formation (90,000
steps/profile) and alignment (10,000 steps/profile), with the previous fresh
formation, cruise, degraded cruise and approach experts. No weights were
averaged. No further training was performed. The selected artifact did not
receive 200,000 steps/profile in each individual expert: that was the total
campaign allocation across six experts, as documented in speed033_training.md.

| Sample | Profile | Speed-0.33 baseline | Selected components | Selected minimum circular clearance (m) |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(table)+'\n\nPromoted: '+str(promoted)+'. The first sample has four starts/profile (91033000..91033003); confirmation has two additional starts/profile (92033000..92033001). These samples do not establish a general success percentage. All outcomes, including failures, remain in the campaign traces.\n'
 text+='\nThe user-discovered severe seed 3624788268 was a TRAINING case, not held-out evidence. In live Gazebo, the selected components completed it at '+str(start)+' s and remained within terminal tolerances through 480 s; the audit found '+str(violations)+' hold violations. The baseline at the same new speed had timed out at 420 s. Additional seed-42 nominal/good Gazebo results are in component_selection_summary.json.\n'
 text+='\nThe promotion gate required no lower per-profile completion in both samples, no observed sampled collisions, at least 0.05 m sampled circular clearance, continuous qualifying hold in successful local-core episodes, and all three Gazebo probes completing. This is not formal collision avoidance or the proposed 100 episodes/profile acceptance benchmark. Hardware command adapters remain absent.\n'
 text+='\nCurrent artifact: '+record['actor_bundle']+'; SHA-256 '+record['sha256']+'. The full six-expert trained artifact and original speed-0.33 baseline remain available separately for reproducibility. No online learning or end-to-end learned phase coordination was introduced.\n'
 (ROOT/'docs/speed033_component_selection.md').write_text(text);print(json.dumps(dict(promoted=promoted,comparisons=comparisons,physical=physical,hard_severe_hold=summary['hard_severe_hold'])),flush=True)

if __name__=='__main__':main()
