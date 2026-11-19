"""Evaluate short stopping before promoting the same learned actor to five meters."""
from pathlib import Path
import json,shutil,subprocess,sys
root=Path(__file__).resolve().parents[2];python=sys.executable
out=root/'experiments/runs/translation_promotion_001';out.mkdir(exist_ok=False)
short=root/'experiments/runs/translation_short_relative_001'
def run(args,log):
 with (out/log).open('w') as stream:subprocess.run([python,*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Completed '+log,flush=True)
def evaluate(folder,seed,episodes,label,task=None):
 args=['scripts/evaluate_formation.py',str(folder),'--episodes',str(episodes),'--seed',str(seed)]
 if task:args+=['--task',str(task)]
 run(args,label+'.log')
 data=json.loads((folder/'evaluation.json').read_text());(out/(label+'.json')).write_text(json.dumps(data,indent=2))
 row=dict(run=str(folder.relative_to(root)),successes=sum(r['reason']=='success' for r in data['results']),episodes=episodes,collisions=sum(r['reason']=='collision' for r in data['results']),mean_goal_error=sum(max(r['goal_errors_m']) for r in data['results'])/episodes)
 print(json.dumps(dict(label=label,**row)),flush=True)
 return row
def checkpoint_folder(checkpoint,manifest):
 folder=out/(manifest.parent.name+'_'+checkpoint.stem);folder.mkdir()
 shutil.copy2(checkpoint,folder/'checkpoint.pt');shutil.copy2(manifest,folder/'manifest.json')
 return folder
choices=[]
for checkpoint in sorted(short.glob('checkpoint_*.pt')):
 folder=checkpoint_folder(checkpoint,short/'manifest.json')
 choices.append(evaluate(folder,31000000,8,folder.name+'_development'))
best=min(choices,key=lambda r:(r['collisions'],-r['successes'],r['mean_goal_error']))
selected=root/best['run']
short_final=evaluate(selected,32000000,40,'short_final')
summary=dict(short_choices=choices,short_selected=best,short_final=short_final,scope='Already formed, aligned headings; ideal localization; good communication')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
if short_final['successes']<30 or short_final['collisions']:
 print('Short-distance promotion gate not met.',flush=True);sys.exit(0)
task=json.loads((root/'configs/tasks/translate_stop_aligned.json').read_text());task['critic_contract']='mission_relative_v1';task['task_name']='translate_stop_five_meter_relative_v1'
task_path=out/'five_meter_task.json';task_path.write_text(json.dumps(task,indent=2))
generalization=evaluate(selected,33000000,8,'five_meter_generalization',task_path)
summary['five_meter_generalization']=generalization
if generalization['successes']>=6 and generalization['collisions']==0:
 final=evaluate(selected,34000000,40,'five_meter_final',task_path)
 summary['five_meter_final']=final
else:
 training=out/'five_meter_training'
 run(['scripts/train_formation.py','--steps','180000','--rollout','512','--epochs','5','--seed','445','--gamma','.999','--gae-lambda','.99','--initial-std','.02','--final-std','.02','--entropy-coef','0','--snapshot-interval','60000','--resume',str(selected/'checkpoint.pt'),'--task',str(task_path),'--reward','configs/training/translation_reward_goal.json','--communication','configs/communication/good.json','--output',str(training)],'five_meter_training.log')
 long_choices=[]
 for checkpoint in sorted(training.glob('checkpoint_*.pt')):
  folder=checkpoint_folder(checkpoint,training/'manifest.json')
  long_choices.append(evaluate(folder,35000000,8,folder.name+'_development'))
 long_best=min(long_choices,key=lambda r:(r['collisions'],-r['successes'],r['mean_goal_error']))
 selected=root/long_best['run']
 summary.update(five_meter_choices=long_choices,five_meter_selected=long_best,five_meter_final=evaluate(selected,36000000,40,'five_meter_final',task_path))
# The short actor may also be evaluated on a five-meter task without changing
# its stored training manifest; exported actor inputs remain the same contract.
run(['scripts/check_actor_export.py',str(selected)],'export_parity.log')
summary['export_parity']='passed';(out/'summary.json').write_text(json.dumps(summary,indent=2))
print('Translation promotion experiment completed.',flush=True)
