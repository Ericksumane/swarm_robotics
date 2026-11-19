"""Run a bounded movement pilot with separate checkpoint selection and tests."""
from pathlib import Path
import json,shutil,subprocess,sys
root=Path(__file__).resolve().parents[2]
out=root/'experiments/runs/translation_aligned_002'
out.mkdir(exist_ok=False)
python=sys.executable
task='configs/tasks/translate_stop_aligned.json'
comm='configs/communication/good.json'
reward='configs/training/translation_reward_aligned.json'
def run(args,log):
 with (out/log).open('w') as stream:
  subprocess.run([python,*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Completed: '+log,flush=True)
run(['scripts/train_formation.py','--steps','180000','--rollout','512','--epochs','5',
 '--seed','441','--sampling-seed','1000441','--gamma','.999','--gae-lambda','.99',
 '--initial-std','.12','--final-std','.03','--std-decay-steps','120000',
 '--entropy-coef','0','--snapshot-interval','60000','--task',task,'--communication',comm,
 '--reward',reward,'--output',str(out/'training')],'training.log')
choices=[]
for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
 folder=out/checkpoint.stem;folder.mkdir()
 shutil.copy2(checkpoint,folder/'checkpoint.pt')
 shutil.copy2(out/'training/manifest.json',folder/'manifest.json')
 run(['scripts/evaluate_formation.py',str(folder),'--episodes','8','--seed','27000000'],folder.name+'_development.log')
 report=json.loads((folder/'evaluation.json').read_text())
 row=dict(run=str(folder.relative_to(root)),successes=sum(r['reason']=='success' for r in report['results']),
  collisions=sum(r['reason']=='collision' for r in report['results']),
  mean_final_goal_error=sum(max(r['goal_errors_m']) for r in report['results'])/8)
 choices.append(row);print(json.dumps(row),flush=True)
best=min(choices,key=lambda r:(r['collisions'],-r['successes'],r['mean_final_goal_error']))
folder=root/best['run']
run(['scripts/evaluate_formation.py',str(folder),'--episodes','40','--seed','28000000'],'final_good.log')
shutil.copy2(folder/'evaluation.json',out/'final_good.json')
summary=dict(scope='Aligned initial headings, formation already established; 5 m and stop only',
 training_steps=180000,development_seed=27000000,final_seed=28000000,choices=choices,selected=best,
 final=json.loads((out/'final_good.json').read_text()))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
run(['scripts/check_actor_export.py',str(folder)],'export_parity.log')
print('Translation pilot completed.',flush=True)
