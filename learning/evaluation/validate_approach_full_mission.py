"""Freeze near-goal selection before paired, fresh full-distance validation."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parents[2]
sources=['approach_recovery_001','approach_recovery_phase_noise_001']
rows=[]
for name in sources:
 report=json.loads((root/'experiments/runs'/name/'summary.json').read_text())
 rows.append(dict(experiment=name,**report['selected']))
best=min(rows,key=lambda r:(r['collisions'],-r['successes'],r['mean_goal_error']))
out=root/'experiments/runs/approach_full_validation_001'
if out.exists():raise FileExistsError(out)
selection=root/'experiments/runs/approach_full_validation_selection.json'
selection.write_text(json.dumps(dict(candidates=rows,selected=best,selection_scope='Near-goal development pool only; final full-mission seeds not inspected',full_validation_seed=53000000,episodes=40,latch_approach=True),indent=2))
out.mkdir()
processes=[];handles=[]
for mode in ('cruise','composed'):
 handle=(out/(mode+'.log')).open('w');handles.append(handle)
 processes.append(subprocess.Popen([sys.executable,'scripts/audit_policy_composition.py','--approach-run',best['run'],'--output',str(out),'--seed','53000000','--episodes','40','--latch-approach','--mode',mode,'--evaluation-role','final'],cwd=root,stdout=handle,stderr=subprocess.STDOUT))
codes=[process.wait() for process in processes]
for handle in handles:handle.close()
if any(code!=0 for code in codes):raise RuntimeError('Full validation worker failed; inspect cruise.log and composed.log.')
base=json.loads((out/'summary_cruise.json').read_text());composed=json.loads((out/'summary_composed.json').read_text())
assert base['provenance']==composed['provenance']
(out/'summary.json').write_text(json.dumps(dict(provenance=base['provenance'],results={**base['results'],**composed['results']}),indent=2))
print('Paired full-distance validation completed.')
