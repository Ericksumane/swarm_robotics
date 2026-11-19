"""Compare the frozen registered model under perfect and good communication."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/control_diagnostics_001';out.mkdir(exist_ok=True)
selected=json.loads((root/'experiments/selected_translation_candidate.json').read_text());bundle_report=json.loads((root/'experiments/runs/approach_full_validation_v2_001/summary.json').read_text())
processes=[];handles=[]
for profile in ['nominal','good']:
 handle=(out/(profile+'.log')).open('w');handles.append(handle)
 processes.append(subprocess.Popen([sys.executable,'scripts/audit_policy_composition.py','--cruise-run',bundle_report['provenance']['cruise'],'--approach-run',bundle_report['provenance']['approach'],'--communication',f'configs/communication/{profile}.json','--output',str(out/profile),'--seed','57000000','--episodes','16','--latch-approach','--approach-distance','.7','--mode','composed'],cwd=root,stdout=handle,stderr=subprocess.STDOUT))
codes=[p.wait() for p in processes]
for handle in handles:handle.close()
if any(codes):raise RuntimeError('Communication diagnostic failed; inspect logs.')
results={profile:json.loads((out/profile/'summary_composed.json').read_text())['results'] for profile in ['nominal','good']}
(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
