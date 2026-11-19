"""Train near-goal recovery and test the full mission using two frozen actors."""
from pathlib import Path
import json,shutil,subprocess,sys,time
root=Path(__file__).resolve().parents[2];python=sys.executable
out=root/'experiments/runs/approach_recovery_001';out.mkdir(exist_ok=False)
def run(args,log):
 with (out/log).open('w') as stream:subprocess.run([python,*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Completed '+log,flush=True)
bank=root/'experiments/runs/approach_capture_001'
for _ in range(180):
 if (bank/'provenance.json').exists():break
 time.sleep(2)
else:raise RuntimeError('Approach capture incomplete')
run(['scripts/train_formation.py','--steps','150000','--rollout','256','--epochs','5','--seed','449','--gamma','.99','--gae-lambda','.95','--initial-std','.02','--final-std','.02','--entropy-coef','0','--snapshot-interval','50000','--resume','experiments/runs/translation_local_goal_001/checkpoint.pt','--task',str(bank/'training_task.json'),'--reward','configs/training/approach_reward.json','--communication','configs/communication/good.json','--output',str(out/'training')],'training.log')
choices=[]
for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
 folder=out/checkpoint.stem;folder.mkdir();shutil.copy2(checkpoint,folder/'checkpoint.pt');shutil.copy2(out/'training/manifest.json',folder/'manifest.json')
 run(['scripts/evaluate_formation.py',str(folder),'--task',str(bank/'heldout_task.json'),'--episodes','12','--seed','51000000'],folder.name+'_approach_development.log')
 data=json.loads((folder/'evaluation.json').read_text())
 row=dict(run=str(folder.relative_to(root)),successes=sum(r['reason']=='success' for r in data['results']),episodes=12,collisions=sum(r['reason']=='collision' for r in data['results']),mean_goal_error=sum(max(r['goal_errors_m']) for r in data['results'])/12)
 choices.append(row);print(json.dumps(row),flush=True)
best=min(choices,key=lambda r:(r['collisions'],-r['successes'],r['mean_goal_error']))
(out/'summary.json').write_text(json.dumps(dict(scope='Near-goal warm resets only; not full five-meter success',choices=choices,selected=best),indent=2))
print('Approach recovery training completed.',flush=True)
