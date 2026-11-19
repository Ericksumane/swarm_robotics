"""Evaluate local availability-based cruise specialists without profile labels."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/local_network_routing_001';out.mkdir(exist_ok=False)
 reference=root/'experiments/runs/geometry_control_001/selected_policy_bundle.pt';source=torch.load(reference,map_location='cpu',weights_only=True)
 def evaluate(bundle,dest,profile,seed,episodes,role):
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle',str(bundle),'--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 development={};candidates=[]
 for checkpoint in sorted((root/'experiments/runs/cruise_network_recovery_001/training').glob('checkpoint_*.pt')):
  state=torch.load(checkpoint,map_location='cpu',weights_only=True);b=copy.deepcopy(source);b['cruise_degraded']={k:state[k] for k in b['cruise']};b['architecture']='local_network_routed_mappo_v3';b['network_routing']=dict(rule='cruise_degraded_when_any_peer_invalid',required_valid_neighbors=2);b['status']='unvalidated_local_network_routing_candidate';b['adaptation']=dict(type='local availability-based frozen cruise specialists',source_bundle_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),degraded_checkpoint=str(checkpoint),degraded_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),original_cruise_and_approach_weights_changed=False)
  bundle=out/(checkpoint.stem+'_bundle.pt');torch.save(b,bundle)
  def dev(profile):return profile,evaluate(bundle,out/(checkpoint.stem+'_'+profile+'_dev'),profile,69000000,4,'development')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(dev,['nominal','good','severe']))
  development[checkpoint.stem]={p:{k:d[k] for k in ['successes','strict_successes','collisions']} for p,d in results.items()}
  if all(results[p]['strict_successes']==4 and results[p]['collisions']==0 for p in ['nominal','good']):
   d=results['severe'];candidates.append(((d['strict_successes'],d['successes'],-sum(max(r['goal_errors_m']) for r in d['results'])),bundle))
 if not candidates:
  (out/'selection.json').write_text(json.dumps(dict(development=development,selected_bundle=None,status='rejected all candidates on nominal/good regression gates'),indent=2));print('No candidate passed regression gates.',flush=True);return
 selected=max(candidates,key=lambda x:x[0])[1]
 selection=dict(development=development,selected_bundle=str(selected),selection_role='Development-only selection; all expert weights frozen before fresh 71M reserved tests')
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 def final(profile):
  d=evaluate(selected,out/(profile+'_final'),profile,71000000,40,'final');return profile,{k:d[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,['nominal','good','moderate','severe']))
 (out/'summary.json').write_text(json.dumps(dict(selection=selection,results=results,episodes=40,seed=71000000),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
