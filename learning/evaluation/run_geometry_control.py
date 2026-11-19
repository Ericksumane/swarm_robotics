"""Matched geometry-cost continuation, development selection, reserved evaluation."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2]
 out=root/'experiments/runs/geometry_control_001';out.mkdir(exist_ok=False)
 source=root/'experiments/runs/translation_heading_precision_001/checkpoint_0120000/checkpoint.pt'
 original=torch.load(root/'experiments/runs/hold_hysteresis_final_001/hysteresis_bundle.pt',map_location='cpu',weights_only=True)
 def training(name):
  run=out/name
  cmd=[sys.executable,'scripts/train_formation.py','--steps','20000','--snapshot-interval','10000','--seed','451','--gamma','.99','--gae-lambda','.95','--actor-lr','.0001','--initial-std','.02','--final-std','.02','--entropy-coef','0','--resume',str(source),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication','configs/communication/nominal.json','--reward',f'configs/training/geometry_{name}.json','--output',str(run)]
  with (out/(name+'.training.log')).open('w') as f:subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return name,run
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:runs=dict(ex.map(training,['control','barrier']))
 bundles={'baseline':root/'experiments/runs/hold_hysteresis_final_001/hysteresis_bundle.pt'}
 def evaluate(bundle,dest,seed,episodes,profile,role):
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 selections={}
 for name,run in runs.items():
  candidates=[]
  for checkpoint in sorted(run.glob('checkpoint_*.pt')):
   state=torch.load(checkpoint,map_location='cpu',weights_only=True)
   b=copy.deepcopy(original);b['cruise']={k:state[k] for k in b['cruise']}
   b['status']='unvalidated_geometry_experiment';b['adaptation']=dict(type='cruise MAPPO continuation only',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),approach_weights_changed=False)
   path=out/(name+'_'+checkpoint.stem+'_bundle.pt');torch.save(b,path)
   data=evaluate(path,out/(name+'_'+checkpoint.stem+'_development'),60000000,8,'nominal','development')
   score=(data['strict_successes'],data['successes'],-data['collisions'],sum(r['formation_compliant_fraction'] for r in data['results']))
   candidates.append((score,path))
  score,path=max(candidates,key=lambda item:item[0]);bundles[name]=path
  selections[name]=dict(bundle=str(path),development_score=list(score))
 (out/'selection.json').write_text(json.dumps(selections,indent=2))
 cells=[(name,profile) for name in bundles for profile in ['nominal','good']]
 def final(cell):
  name,profile=cell;data=evaluate(bundles[name],out/(name+'_'+profile+'_final'),61000000,40,profile,'final')
  return name+'_'+profile,{k:data[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,cells))
 (out/'summary.json').write_text(json.dumps(dict(training_steps_per_branch=20000,final_seed=61000000,episodes_per_cell=40,selection=selections,results=results),indent=2))
 print(json.dumps(results),flush=True)

if __name__=='__main__':main()
