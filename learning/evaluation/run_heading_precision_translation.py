"""Fine-tune the local-goal actor for full distance with independent tests."""
from pathlib import Path
import json,shutil,subprocess,sys
root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/translation_heading_precision_001';out.mkdir(exist_ok=False)
python=sys.executable
def run(args,log):
 with (out/log).open('w') as stream:subprocess.run([python,*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Completed '+log,flush=True)
run(['scripts/train_formation.py','--steps','120000','--rollout','256','--epochs','5','--seed','448','--gamma','.99','--gae-lambda','.95','--initial-std','.02','--final-std','.02','--entropy-coef','0','--snapshot-interval','40000','--resume','experiments/runs/translation_local_goal_001/checkpoint.pt','--task','configs/tasks/translate_stop_local_goal.json','--reward','configs/training/translation_reward_heading_precision.json','--communication','configs/communication/good.json','--output',str(out/'training')],'training.log')
choices=[]
for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
 folder=out/checkpoint.stem;folder.mkdir();shutil.copy2(checkpoint,folder/'checkpoint.pt');shutil.copy2(out/'training/manifest.json',folder/'manifest.json')
 run(['scripts/evaluate_formation.py',str(folder),'--episodes','8','--seed','44000000'],folder.name+'_development.log')
 data=json.loads((folder/'evaluation.json').read_text())
 row=dict(run=str(folder.relative_to(root)),successes=sum(r['reason']=='success' for r in data['results']),collisions=sum(r['reason']=='collision' for r in data['results']),mean_goal_error=sum(max(r['goal_errors_m']) for r in data['results'])/8)
 choices.append(row);print(json.dumps(row),flush=True)
best=min(choices,key=lambda r:(r['collisions'],-r['successes'],r['mean_goal_error']));selected=root/best['run']
run(['scripts/evaluate_formation.py',str(selected),'--episodes','40','--seed','45000000'],'final_good.log')
data=json.loads((selected/'evaluation.json').read_text());(out/'final_good.json').write_text(json.dumps(data,indent=2))
run(['scripts/check_actor_export.py',str(selected)],'export_parity.log')
(out/'summary.json').write_text(json.dumps(dict(choices=choices,selected=best,final=data,export_parity='passed'),indent=2))
print('Local-goal five-meter training completed.',flush=True)
