"""Select tighter startup acquisition before fresh physical/full tests."""
from pathlib import Path
import concurrent.futures,json,subprocess,sys

def main():
 root=Path(__file__).resolve().parents[2];training=root/'experiments/runs/dispersed_hard_recovery_001';out=root/'experiments/runs/dispersed_hard_validation_001';out.mkdir(exist_ok=False)
 if not (training/'training/training_summary.json').exists():raise RuntimeError('Hard acquisition training is not complete.')
 first=json.loads((root/'experiments/runs/dispersed_recovery_validation_001/selection.json').read_text())
 def component(checkpoint,dest,profile,seed,episodes,role,precision=False):
  task='configs/tasks/dispersed_precision_evaluation.json' if precision else 'configs/tasks/formation_relational_broad_evaluation.json'
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_startup_component.py','--checkpoint',str(checkpoint),'--task',task,'--reward',str(training/'training_reward.json'),'--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 development={};candidates=[]
 checkpoints=[('reference',Path(first['acquisition']['checkpoint'])),*[(p.stem,p) for p in sorted((training/'training').glob('checkpoint_*.pt'))]]
 for name,checkpoint in checkpoints:
  def dev(profile):return profile,component(checkpoint,out/(name+'_'+profile+'_dev'),profile,77000000,8,'development',precision=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(dev,['nominal','good']))
  score=(-sum(d['collisions'] for d in results.values()),sum(d['successes'] for d in results.values()),-sum(r['elapsed_time_s'] for d in results.values() for r in d['results']))
  development[name]=dict(score=list(score),results={p:{k:d[k] for k in ['successes','collisions']} for p,d in results.items()})
  candidates.append((score,checkpoint,out/(name+'_nominal_dev')/'actor.pt',name))
 score,checkpoint,actor,name=max(candidates,key=lambda x:x[0]);selection=dict(checkpoint=str(checkpoint),actor=str(actor),name=name,score=list(score),development=development,alignment_actor=first['alignment']['actor'],selection_role='77M development with stricter 0.05 m geometry; reserved physical criterion remains 0.1 m')
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 cells=[(version,profile) for version in ['reference','candidate'] for profile in ['nominal','good']]
 def final(cell):
  version,profile=cell;path=first['acquisition']['checkpoint'] if version=='reference' else selection['checkpoint'];d=component(path,out/(version+'_'+profile+'_final'),profile,78000000,40,'final');return version+'_'+profile,{k:d[k] for k in ['successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,cells))
 (out/'component_summary.json').write_text(json.dumps(dict(selection=selection,results=results),indent=2))
 if any(results['candidate_'+p]['collisions']>results['reference_'+p]['collisions'] for p in ['nominal','good']):
  (out/'summary.json').write_text(json.dumps(dict(status='rejected_component_safety_regression',selection=selection,components=results,full_mission={}),indent=2))
  print('Rejected on paired collision regression; integration skipped.',flush=True);return
 def full(cell):
  profile,limit=cell;dest=out/('full_'+profile+'_'+str(limit)+'s')
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_startup_mission.py','--formation',selection['actor'],'--alignment',selection['alignment_actor'],'--translation','experiments/runs/local_network_routing_001/selected_policy_bundle.pt','--task','configs/tasks/dispersed_full_tight_start.json','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed','79000000','--episodes','20','--time-limit',str(limit)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((dest/'evaluation.json').read_text());return profile+'_'+str(limit)+'s',{k:d[k] for k in ['successes','strict_successes','collisions','acquired','aligned','started_move']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:integrated=dict(ex.map(full,[(p,l) for l in [120,240] for p in ['nominal','good']]))
 (out/'summary.json').write_text(json.dumps(dict(selection=selection,components=results,full_mission=integrated,component_episodes=40,full_mission_episodes=20),indent=2));print(json.dumps(dict(components=results,full_mission=integrated)),flush=True)

if __name__=='__main__':main()
