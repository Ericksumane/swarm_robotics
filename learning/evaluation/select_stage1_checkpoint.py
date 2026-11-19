"""Select formation checkpoints on validation seeds, then compare independently."""
import argparse,json,shutil,statistics,subprocess,sys
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
 root=Path(__file__).resolve().parents[2];run=Path(args.run).resolve()
 def evaluate(folder,seed,count,mode='policy',task=None):
  command=[sys.executable,str(root/'scripts/evaluate_formation.py'),str(folder),'--seed',str(seed),'--episodes',str(count),'--mode',mode]
  if task:command+=['--task',task]
  with (folder/f'stage1_{mode}_{seed}.log').open('w') as log:
   subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  return json.loads((folder/'evaluation.json').read_text())
 def metrics(report):
  rows=report['results']
  return dict(success_rate=report['success_rate'],collisions=sum(r['reason']=='collision' for r in rows),mean_rmse_m=statistics.mean(r['mean_rmse_m'] for r in rows),episodes=len(rows))
 candidates=[];seen_steps=set()
 import torch
 for checkpoint in sorted(run.glob('checkpoint_*.pt'))+[run/'checkpoint.pt']:
  metadata=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if metadata['step'] in seen_steps:continue
  seen_steps.add(metadata['step'])
  folder=run/f"validation_{metadata['step']}";folder.mkdir()
  shutil.copy2(checkpoint,folder/'checkpoint.pt');shutil.copy2(run/'manifest.json',folder/'manifest.json')
  result=metrics(evaluate(folder,700100,8));result.update(step=metadata['step'],path=str(folder))
  candidates.append(result);print('Validation:',json.dumps(result),flush=True)
 best=max(candidates,key=lambda x:(x['success_rate'],-x['collisions'],-x['mean_rmse_m']))
 selected=run/'selected';selected.mkdir()
 shutil.copy2(Path(best['path'])/'checkpoint.pt',selected/'checkpoint.pt');shutil.copy2(run/'manifest.json',selected/'manifest.json')
 reports={}
 reports['selected_nontrivial']=evaluate(selected,800100,20)
 # Preserve acquisition replay before subsequent evaluation modes.
 shutil.copy2(selected/'trajectory.json',selected/'acquisition_trajectory.json')
 reports['stationary_nontrivial']=evaluate(selected,800100,20,mode='stationary')
 reports['selected_broad']=evaluate(selected,800100,10,task='configs/tasks/formation_v1.json') if False else None
 # Broad starts must retain the new observation contract.
 manifest=json.loads((run/'manifest.json').read_text())
 broad=json.loads((root/'configs/tasks/formation_v1.json').read_text());broad.update(observation_contract=manifest['contract'],task_name='formation_geometry_broad_v2')
 broad_path=run/'broad_evaluation_task.json';broad_path.write_text(json.dumps(broad,indent=2))
 reports['selected_broad']=evaluate(selected,800100,10,task=str(broad_path))
 previous=root/'experiments/runs/formation_diagnostic_001/selected'
 reports['previous_nontrivial']=evaluate(previous,800100,20,task='configs/tasks/formation_diagnostic.json')
 results={name:metrics(report) for name,report in reports.items()}
 (run/'comparison.json').write_text(json.dumps(dict(candidates=candidates,selected=best,metrics=results,reports=reports),indent=2))
 shutil.copy2(selected/'acquisition_trajectory.json',selected/'trajectory.json')
 subprocess.run([sys.executable,str(root/'scripts/create_replay.py'),str(selected)],cwd=root,check=True)
 subprocess.run([sys.executable,str(root/'scripts/check_actor_export.py'),str(selected)],cwd=root,check=True)
 lines=['# Formation stage-one geometry experiment','',
  '| Evaluation | Success rate | Collisions | Mean RMSE (m) |',
  '|---|---:|---:|---:|']
 for name,result in results.items():
  lines.append(f"| {name} | {result['success_rate']:.1%} ({result['episodes']} episodes) | {result['collisions']} | {result['mean_rmse_m']:.4f} |")
 lines+=['',f"Selected checkpoint: {best['step']} steps.",
  'Selection used seeds 700100–700107; comparison used separate seeds 800100 onward.',
  'No five-meter translation or hardware validation is implied.',
  'The selected replay shows nontrivial-start formation acquisition, not broad starts.']
 (root/'docs/formation_stage1_geometry_results.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
