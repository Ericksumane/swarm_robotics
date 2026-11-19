"""Finish focused 100k arrival training with unchanged full-mission gates."""
from pathlib import Path
import concurrent.futures,hashlib,json,math,os,subprocess,sys,time
import torch
from swarm_core.mission_spec import completion_task
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/runs/formation_clearance_100k_001';PROFILES=['nominal','good','moderate','severe']
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def report_ready(path):
 try:
  d=read(path);return d if set(d['profiles'])==set(PROFILES) and all(len(v)==4 for v in d['profiles'].values()) else None
 except (FileNotFoundError,json.JSONDecodeError):return None

def main():
 status=BASE/'pipeline_status.json';write(status,dict(status='training',total_requested_environment_steps=100000))
 deadline=time.monotonic()+3600
 while True:
  try:progress=read(BASE/'training/progress.json')
  except FileNotFoundError:progress={}
  if progress.get('status')=='training_complete':break
  pid=read(BASE/'process.json')['pid'];proc=Path('/proc')/str(pid)/'stat'
  if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Trainer ended before its exact budget; inspect training.log.')
  if time.monotonic()>deadline:raise RuntimeError('Training wait exceeded; do not promote an unfinished model.')
  time.sleep(10)
 assert sum(progress['steps_per_profile'].values())==100000 and all(v==25000 for v in progress['steps_per_profile'].values())
 baseline=torch.load(BASE/'baseline_bundle.pt',map_location='cpu',weights_only=True);path=BASE/'training/candidate_bundle.pt';candidate=torch.load(path,map_location='cpu',weights_only=True)
 def experts(b):return dict(history=b['formation'],fresh=b['formation']['fresh_actor'],alignment=b['alignment'],cruise=b['translation']['cruise'],degraded=b['translation']['cruise_degraded'],approach=b['translation']['approach'])
 a,z=experts(baseline),experts(candidate);changes={k:any(not torch.equal(v['actor'][n],a[k]['actor'][n]) for n in v['actor'] if 'logstd' not in n) for k,v in z.items()};assert changes['history'] and not any(v for k,v in changes.items() if k!='history')
 write(status,dict(status='evaluating',completed_steps_per_profile=progress['steps_per_profile'],changed_experts=changes))
 record=dict(actor_bundle=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='formation_clearance_candidate_pending_evaluation',weights_changed=True,report='docs/formation_clearance_100k_results.md',hardware_approved=False,training_total_environment_steps=100000,training_steps_per_profile=progress['steps_per_profile'],trained_experts=['formation_history'])
 write(BASE/'candidate.json',record)
 def local_eval(label):
  out=BASE/('evaluation_'+label)
  if label=='baseline':
   for _ in range(90):
    d=report_ready(out/'evaluation.json')
    if d:return label,d
    time.sleep(10)
   raise RuntimeError('Baseline evaluation incomplete.')
  with (BASE/('evaluation_'+label+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/validate_local_runtime.py','--bundle',str(path),'--output',str(out),'--seed','94055000','--episodes','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  d=report_ready(out/'evaluation.json');assert d;return label,d
 def physical(case):
  label,profile,seed=case;limit=420 if profile=='severe' else 300;duration=480 if profile=='severe' else 360;out=BASE/('gazebo_'+label+'_candidate')
  with (BASE/('gazebo_'+label+'_candidate.log')).open('w') as log:subprocess.run(['bash','scripts/run_live_gazebo.sh','--startup-bundle',str(path),'--communication',profile,'--seed',str(seed),'--time-limit',str(limit),'--duration',str(duration),'--headless','--fast','--keep-controlling','--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  d=read(out/'summary.json');first=d['final_policy']['first_success_s'];minimum=math.inf;violations=0;hold_samples=0;tol=completion_task(candidate['task'],profile)['mission']['formation_tolerance_m']
  for line in (out/'control.jsonl').open():
   f=json.loads(line);p=f['policy'];minimum=min(minimum,p['minimum_clearance_m'])
   if first is not None and p['time_s']>=first-1e-8:
    hold_samples+=1;violations+=int(any(e is None or e>.2 for e in p['goal_errors_m']) or p['max_pair_error_m']>tol or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']))
  return label,dict(profile=profile,seed=seed,completed=d['final_policy']['mission_completed'],first_success_s=first,minimum_clearance_m=minimum,hold_samples=hold_samples,hold_violations=violations,goal_errors_m=d['final_policy']['goal_errors_m'],reason=d['reason'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(local_eval,label) for label in ['baseline','candidate']];reports=dict(f.result() for f in futures)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:physical_results=dict(pool.map(physical,[('nominal_close','nominal',613953391),('good_close','good',917443879),('arrival_regression','nominal',47849202)]))
 comparison={};safe=True;improved=False;audit=[]
 for profile in PROFILES:
  b=reports['baseline']['profiles'][profile];c=reports['candidate']['profiles'][profile];assert [r['seed'] for r in b]==[r['seed'] for r in c]
  preserved=all(x['reason']!='success' or y['reason']=='success' for x,y in zip(b,c));improved=improved or any(x['reason']!='success' and y['reason']=='success' for x,y in zip(b,c));collisions=sum(r['reason']=='collision' for r in c);minimum=min(r['minimum_clearance_m'] for r in c);clearance_preserved=all(y['minimum_clearance_m']>=x['minimum_clearance_m']-.01 for x,y in zip(b,c));safe=safe and preserved and collisions==0 and minimum>=.05 and clearance_preserved
  tol=completion_task(candidate['task'],profile)['mission']['formation_tolerance_m']
  for row in c:
   frames=read(BASE/'evaluation_candidate'/(profile+'_'+str(row['seed'])+'.json'));held=[f for f in frames if row['first_success_s'] is not None and f['time_s']>=row['first_success_s']-1e-8];violations=sum(any(e is None or e>.2 for e in f['goal_errors_m']) or f['pair_error_m']>tol or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']) for f in held)
   if row['reason']=='success':safe=safe and len(held)>=600 and violations==0
   audit.append(dict(profile=profile,seed=row['seed'],success=row['reason']=='success',hold_samples=len(held),hold_violations=violations))
  comparison[profile]=dict(baseline_successes=sum(r['reason']=='success' for r in b),candidate_successes=sum(r['reason']=='success' for r in c),runs=4,all_baseline_success_cases_preserved=preserved,candidate_collisions=collisions,candidate_minimum_clearance_m=minimum,clearance_preserved=clearance_preserved)
 for value in physical_results.values():safe=safe and value['completed'] and value['minimum_clearance_m']>=.05 and value['hold_samples']>=600 and value['hold_violations']==0
 known_clearances=[physical_results[k]['minimum_clearance_m'] for k in ('nominal_close','good_close')];known_improved=all(v>=.15 for v in known_clearances) and any(v>=.20 for v in known_clearances)
 accepted=bool(safe and known_improved);current=read(ROOT/'experiments/selected_delivery_candidate.json');baseline_hash=hashlib.sha256((BASE/'baseline_bundle.pt').read_bytes()).hexdigest();promoted=bool(accepted and current['sha256']==baseline_hash)
 record.update(status='formation_clearance_selected_pending_final_acceptance' if promoted else 'formation_clearance_not_promoted',changed_experts=changes);write(BASE/'candidate.json',record)
 if promoted:
  write(ROOT/'experiments/selected_delivery_candidate.json',record);p=ROOT/'configs/deployment/delivery_candidate.json';d=read(p);d.update(actor_bundle=record['actor_bundle'],actor_bundle_sha256=record['sha256'],status=record['status'],release_id='formation_clearance_100k_001',policy_final_acceptance_met=False);write(p,d)
 result=dict(status='complete',promoted=promoted,accepted=accepted,training_steps_per_profile=progress['steps_per_profile'],total_environment_steps=100000,changed_experts=changes,comparison=comparison,physical=physical_results,known_clearance_gate_met=known_improved,hold_audit=audit,candidate=record,scope='Finite new-start paired validation and known training-case regression probes; not final 100/profile acceptance or hardware approval.')
 write(BASE/'summary.json',result);write(status,result)
 text="# Formation clearance training results\n\nExactly 100,000 environment steps, 25,000/profile. Only history formation changed; the other five neural experts, deployment gate and task criteria were preserved. See formation_clearance_100k.md for the predeclared selection gate.\n\nPromoted: "+str(promoted)+"\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n\nThis finite paired evaluation is not hardware approval or the final 100 runs/profile benchmark. Sampled circular clearances are not continuous physical contact measurements.\n"
 (ROOT/'docs/formation_clearance_100k_results.md').write_text(text);print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:write(BASE/'pipeline_status.json',dict(status='failed',error=str(error)));raise
