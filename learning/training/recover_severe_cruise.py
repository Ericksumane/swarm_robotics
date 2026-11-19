"""Cruise-only network recovery after diagnosing pre-approach lateral drift."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/cruise_network_recovery_001';out.mkdir(exist_ok=False)
 source_path=root/'experiments/runs/geometry_control_001/selected_policy_bundle.pt';source=torch.load(source_path,map_location='cpu',weights_only=True)
 resume=root/'experiments/runs/geometry_control_001/control/checkpoint_0010240.pt'
 with (out/'training.log').open('w') as log:subprocess.run([sys.executable,'scripts/train_formation.py','--steps','20000','--snapshot-interval','10000','--seed','454','--gamma','.99','--actor-lr','.0001','--initial-std','.02','--final-std','.02','--entropy-coef','0','--resume',str(resume),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--reward','configs/training/geometry_control.json','--communication-mixture','configs/training/cruise_communication_mixture.json','--output',str(out/'training')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 def evaluate(bundle,dest,profile,seed,episodes,role):
  with dest.with_suffix('.log').open('w') as log:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 candidates=[];development={}
 for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
  state=torch.load(checkpoint,map_location='cpu',weights_only=True);b=copy.deepcopy(source);b['cruise']={k:state[k] for k in b['cruise']};b['status']='unvalidated_cruise_network_candidate';b['adaptation']=dict(type='cruise-only network mixture MAPPO continuation',source_bundle_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),checkpoint=str(checkpoint),approach_weights_changed=False)
  path=out/(checkpoint.stem+'_bundle.pt');torch.save(b,path)
  def dev(profile):return profile,evaluate(path,out/(checkpoint.stem+'_'+profile+'_dev'),profile,69000000,4,'development')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(dev,['nominal','good','severe']))
  development[checkpoint.stem]={p:{k:d[k] for k in ['successes','strict_successes','collisions']} for p,d in results.items()}
  if all(results[p]['strict_successes']==4 and results[p]['collisions']==0 for p in ['nominal','good']):
   severe=results['severe'];candidates.append(((severe['strict_successes'],severe['successes'],-sum(max(r['goal_errors_m']) for r in severe['results'])),path))
 selected=max(candidates,key=lambda x:x[0])[1] if candidates else source_path
 selection=dict(selected_bundle=str(selected),development=development,role='Frozen development selection; reserved evaluation is not used for model choice')
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 def final(profile):
  d=evaluate(selected,out/(profile+'_final'),profile,70000000,40,'final');return profile,{k:d[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,['nominal','good','moderate','severe']))
 (out/'summary.json').write_text(json.dumps(dict(selection=selection,results=results,episodes=40,seed=70000000),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
