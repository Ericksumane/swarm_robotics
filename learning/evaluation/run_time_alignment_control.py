"""One-variable frozen-policy comparison under nominal and good communication."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/time_alignment_control_001';out.mkdir(exist_ok=False)
 source=root/'experiments/runs/approach_full_validation_v2_001/policy_bundle.pt';legacy=torch.load(source,map_location='cpu',weights_only=True)
 aligned=dict(legacy);aligned.update(contract='mission_local_goal_time_aligned_v4',status='unvalidated_observation_adapter',adaptation=dict(source_contract=legacy['contract'],source_bundle_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),weights_changed=False,type='valid received peer pose extrapolated by original message age'))
 adapted=out/'aligned_bundle.pt';torch.save(aligned,adapted)
 for name in ('cruise','approach'):
  assert all(torch.equal(value,aligned[name]['actor'][key]) for key,value in legacy[name]['actor'].items())
 cells=[(version,profile) for profile in ('nominal','good') for version in ('legacy','aligned')]
 def run(cell):
  version,profile=cell;name=version+'_'+profile;bundle=source if version=='legacy' else adapted;task='configs/tasks/translate_stop_local_goal.json' if version=='legacy' else 'configs/tasks/translate_stop_time_aligned.json'
  with (out/(name+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task',task,'--communication',f'configs/communication/{profile}.json','--output',str(out/name),'--seed','57000000','--episodes','16'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  data=json.loads((out/name/'evaluation.json').read_text());return name,{key:data[key] for key in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:results=dict(executor.map(run,cells))
 expected={'legacy_nominal':16,'legacy_good':5}
 for name,count in expected.items():
  if results[name]['successes']!=count:raise RuntimeError('Legacy compatibility check failed: '+name)
 (out/'summary.json').write_text(json.dumps(dict(seed=57000000,episodes=16,legacy_compatibility_passed=True,results=results),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
