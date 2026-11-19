"""Run the requested balanced training and an explicit finite regression gate."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/speed033_campaign_001'

def atomic(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 started=time.monotonic();status=BASE/'pipeline_status.json';atomic(status,dict(status='training',requested_steps_per_profile=200000,profiles=['nominal','good','moderate','severe']))
 command=[sys.executable,'learning/training/train_speed033.py','--device','cuda','--output',str(BASE/'training')]
 if (BASE/'training/manifest.json').exists():command.append('--resume')
 with (BASE/'training.log').open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 progress=json.loads((BASE/'training/progress.json').read_text());assert progress['status']=='training_complete' and all(v==200000 for v in progress['steps_per_profile'].values())
 atomic(status,dict(status='evaluating',completed_steps_per_profile=progress['steps_per_profile'],training_candidate=str(BASE/'training/candidate_bundle.pt')))
 def evaluate(name):
  bundle=BASE/('baseline_bundle.pt' if name=='baseline' else 'training/candidate_bundle.pt');out=BASE/('evaluation_'+name)
  if not (out/'evaluation.json').exists():
   if out.exists():raise RuntimeError('Partial evaluation exists: preserve and choose a new output before retrying.')
   with (BASE/('evaluation_'+name+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/validate_local_runtime.py','--bundle',str(bundle),'--output',str(out),'--seed','91033000','--episodes','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  return name,json.loads((out/'evaluation.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:reports=dict(pool.map(evaluate,['baseline','trained']))
 comparison={};safe=True;improved=False;rows=[]
 for profile in ['nominal','good','moderate','severe']:
  scores={}
  for name,report in reports.items():
   runs=report['profiles'][profile];assert len(runs)==4
   scores[name]=dict(successes=sum(r['reason']=='success' for r in runs),collisions=sum(r['reason']=='collision' for r in runs),minimum_clearance_m=min(r['minimum_clearance_m'] for r in runs),reasons=[r['reason'] for r in runs])
  b,t=scores['baseline'],scores['trained'];safe=safe and t['collisions']==0 and t['successes']>=b['successes'] and t['minimum_clearance_m']>=.05;improved=improved or t['successes']>b['successes'];comparison[profile]=scores
  rows.append('| '+profile+' | '+str(b['successes'])+'/4 | '+str(t['successes'])+'/4 | '+str(t['collisions'])+' | '+format(t['minimum_clearance_m'],'.3f')+' |')
 accepted=bool(safe and (improved or all(c['trained']['successes']==4 for c in comparison.values())))
 trained=BASE/'training/candidate_bundle.pt';digest=hashlib.sha256(trained.read_bytes()).hexdigest();baseline_digest=hashlib.sha256((BASE/'baseline_bundle.pt').read_bytes()).hexdigest()
 record=dict(status='trained_candidate_passed_small_regression_gate' if accepted else 'trained_candidate_not_promoted',actor_bundle=str(trained.relative_to(ROOT)),sha256=digest,weights_changed=True,report='docs/speed033_training_results.md',hardware_approved=False,training_steps_per_profile=progress['steps_per_profile'])
 atomic(BASE/'trained_candidate.json',record)
 # Preserve any newer user-selected candidate instead of overwriting it.
 current=json.loads((ROOT/'experiments/selected_delivery_candidate.json').read_text())
 if accepted and current['sha256']==baseline_digest:
  atomic(ROOT/'experiments/selected_delivery_candidate.json',record)
  manifest_path=ROOT/'configs/deployment/delivery_candidate.json';manifest=json.loads(manifest_path.read_text());manifest.update(actor_bundle=record['actor_bundle'],actor_bundle_sha256=digest,status=record['status'],policy_final_acceptance_met=False,release_id='speed033_trained_candidate_001');atomic(manifest_path,manifest)
  promoted=True
 else:promoted=False
 summary=dict(status='complete',completed_steps_per_profile=progress['steps_per_profile'],total_environment_steps=sum(progress['steps_per_profile'].values()),comparison=comparison,passed_small_regression_gate=accepted,promoted=promoted,trained_candidate=record,elapsed_s=time.monotonic()-started,scope='Finite independent-local-runtime tests, four new starts per communication profile, ideal synthetic physical plant. Not final 100/profile acceptance or hardware validation.')
 atomic(BASE/'summary.json',summary);atomic(status,summary)
 text='''# Speed 0.33 training results

All six MAPPO experts were adapted with 200,000 environment steps per
communication profile: nominal, good, moderate and severe. Total 800,000
three-robot environment steps (2,400,000 robot transitions). Smoke and evaluation
steps are excluded. Training used the plan and source hashes in the campaign.

The complete trained candidate was frozen before four new initializations per
profile (seed 91033000..91033003). Baseline and trained models use the same
0.33/0.165 m/s standard. Tests execute three independent local runtime cores
with synthetic impaired messages and a Python physical plant, including a
60-second observation after success. They are not Gazebo or hardware trials.

| Profile | Baseline success | Trained success | Trained collisions | Trained minimum circular clearance (m) |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'\n\nPromoted into the current live registry: '+str(promoted)+'.\n'
 text+='Training completion is not proof of improvement. The gate requires no sampled collisions, no lower per-profile completion, at least 0.05 m sampled clearance, and either improved completion or all 16 successes. This is a small development gate, not the proposed 100 episodes per profile acceptance benchmark.\n'
 text+='\nThe trained candidate remains at '+str(trained.relative_to(ROOT))+', SHA-256 '+digest+'. Previous weights and the new-speed baseline are preserved. Routing, phase coordination and stop supervisors remain programmed; the campaign did not learn an end-to-end coordinator. Hardware approval remains false.\n'
 (ROOT/'docs/speed033_training_results.md').write_text(text);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  atomic(BASE/'pipeline_status.json',dict(status='failed',error=str(error),resume='Inspect logs; training supports --resume. Do not count an unfinished run as complete.'));raise
