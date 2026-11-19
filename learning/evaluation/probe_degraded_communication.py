"""Exploratory frozen-policy probe after nominal/good gates passed."""
from pathlib import Path
import concurrent.futures,json,subprocess,sys

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/communication_probe_001';out.mkdir(exist_ok=False)
 reference=root/'experiments/runs/geometry_control_001'
 for p in ['nominal','good']:
  d=json.loads((reference/('control_'+p+'_final')/'evaluation.json').read_text())
  if d['successes']!=40 or d['strict_successes']!=40 or d['collisions']!=0:raise RuntimeError('Nominal/good gate not satisfied.')
 selection=json.loads((reference/'selection.json').read_text());bundle=selection['control']['bundle']
 def run(profile):
  with (out/(profile+'.log')).open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',bundle,'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(out/profile),'--seed','62000000','--episodes','12','--role','development'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((out/profile/'evaluation.json').read_text());return profile,{k:d[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(run,['moderate','severe']))
 summary=dict(role='exploratory diagnostic, not robustness certification',seed=62000000,episodes=12,weights_changed=False,results=results)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
