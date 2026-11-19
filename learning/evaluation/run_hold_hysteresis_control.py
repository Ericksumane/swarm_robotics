"""One-parameter HOLD entry comparison with frozen actors and physical scoring."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/hold_hysteresis_control_001';out.mkdir(exist_ok=False)
 source=root/'experiments/runs/approach_full_validation_v2_001/policy_bundle.pt';legacy=torch.load(source,map_location='cpu',weights_only=True)
 candidate=dict(legacy);candidate.update(status='unvalidated_task_protocol',mission_protocol=dict(hold_entry_tolerance_m=.15,hold_exit_tolerance_m=.2,physical_success_tolerance_m=.2),adaptation=dict(source_bundle_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),weights_changed=False,type='HOLD entry .15 m; exit and physical goal criterion .2 m unchanged'))
 adapted=out/'hysteresis_bundle.pt';torch.save(candidate,adapted)
 cells=[(version,profile) for profile in ('nominal','good') for version in ('legacy','hysteresis')]
 def run(cell):
  version,profile=cell;name=version+'_'+profile;bundle=source if version=='legacy' else adapted;task='configs/tasks/translate_stop_local_goal.json' if version=='legacy' else 'configs/tasks/translate_stop_hold_hysteresis.json'
  with (out/(name+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task',task,'--communication',f'configs/communication/{profile}.json','--output',str(out/name),'--seed','57000000','--episodes','16'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  data=json.loads((out/name/'evaluation.json').read_text());return name,{key:data[key] for key in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:results=dict(executor.map(run,cells))
 if results['legacy_nominal']['successes']!=16 or results['legacy_good']['successes']!=5:raise RuntimeError('Legacy physical-score equivalence failed.')
 (out/'summary.json').write_text(json.dumps(dict(seed=57000000,episodes=16,legacy_compatibility_passed=True,results=results),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
