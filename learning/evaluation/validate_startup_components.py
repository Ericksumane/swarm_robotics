"""Select startup components before reserved acquisition/alignment/full tests."""
from pathlib import Path
import concurrent.futures,json,subprocess,sys

def main():
 root=Path(__file__).resolve().parents[2];training=root/'experiments/runs/dispersed_recovery_001';out=root/'experiments/runs/dispersed_recovery_validation_001';out.mkdir(exist_ok=False)
 if not (training/'training_summary.json').exists():raise RuntimeError('Both component trainings must finish first.')
 def component(checkpoint,mode,profile,dest,seed,episodes,role):
  task='configs/tasks/alignment_v1.json' if mode=='alignment' else 'configs/tasks/formation_relational_broad_evaluation.json';reward='configs/training/alignment_reward.json' if mode=='alignment' else 'configs/training/dispersed_acquisition_reward.json'
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_startup_component.py','--checkpoint',str(checkpoint),'--task',task,'--reward',reward,'--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed',str(seed),'--episodes',str(episodes),'--role',role],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return json.loads((dest/'evaluation.json').read_text())
 selection={}
 for mode in ['acquisition','alignment']:
  candidates=[];development={}
  for checkpoint in sorted((training/mode).glob('checkpoint_*.pt')):
   def dev(profile):return profile,component(checkpoint,mode,profile,out/(mode+'_'+checkpoint.stem+'_'+profile+'_dev'),72000000,8,'development')
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(dev,['nominal','good']))
   score=(-sum(d['collisions'] for d in results.values()),sum(d['successes'] for d in results.values()),-sum(r['elapsed_time_s'] for d in results.values() for r in d['results']))
   actor=out/(mode+'_'+checkpoint.stem+'_nominal_dev')/'actor.pt';candidates.append((score,checkpoint,actor));development[checkpoint.stem]=dict(score=list(score),results={p:{k:d[k] for k in ['successes','collisions']} for p,d in results.items()})
  score,checkpoint,actor=max(candidates,key=lambda x:x[0]);selection[mode]=dict(checkpoint=str(checkpoint),actor=str(actor),development=development,selected_score=list(score))
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 cells=[(name,profile) for name in ['baseline_acquisition','acquisition','alignment'] for profile in ['nominal','good']]
 def final(cell):
  name,profile=cell;mode='alignment' if name=='alignment' else 'acquisition';checkpoint=root/'experiments/runs/extended_horizon_001/continued_final/checkpoint.pt' if name=='baseline_acquisition' else Path(selection[mode]['checkpoint'])
  d=component(checkpoint,mode,profile,out/(name+'_'+profile+'_final'),74000000 if mode=='alignment' else 73000000,40,'final');return name+'_'+profile,{k:d[k] for k in ['successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(final,cells))
 (out/'component_summary.json').write_text(json.dumps(dict(selection=selection,results=results),indent=2))
 def integrated(cell):
  profile,limit=cell;dest=out/('full_'+profile+'_'+str(limit)+'s')
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_startup_mission.py','--formation',selection['acquisition']['actor'],'--alignment',selection['alignment']['actor'],'--translation','experiments/runs/local_network_routing_001/selected_policy_bundle.pt','--communication',f'configs/communication/{profile}.json','--output',str(dest),'--seed','75000000','--episodes','20','--time-limit',str(limit)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((dest/'evaluation.json').read_text());return profile+'_'+str(limit)+'s',{k:d[k] for k in ['successes','strict_successes','collisions','acquired','aligned','started_move']}
 # Primary 120 s test plus explicitly separate 240 s latency diagnostic.
 cells=[(p,limit) for limit in [120,240] for p in ['nominal','good']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:full=dict(ex.map(integrated,cells))
 (out/'summary.json').write_text(json.dumps(dict(selection=selection,components=results,full_mission=full,component_episodes=40,full_mission_episodes=20),indent=2));print(json.dumps(dict(components=results,full_mission=full)),flush=True)

if __name__=='__main__':main()
