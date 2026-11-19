"""Paired nominal/good communication training with scheduled exploration."""
import argparse,json,statistics,subprocess,sys,shutil
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);cli=parser.parse_args()
 root=Path(__file__).resolve().parents[2];out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
 initial=root/'experiments/runs/critic_ab_001/initial_actor_43.pt'
 configs={'nominal':'configs/communication/nominal.json','good':'configs/communication/good.json'}
 results=[]
 def command(args,path):
  with path.open('w') as log:subprocess.run([sys.executable]+args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 protocol=dict(seed=43,sampling_seed=1000043,steps_per_model=50000,std_start=.1,std_final=.02,std_decay_steps=20000,evaluation_seed=1000100,evaluation_episodes=20,task='configs/tasks/formation_relative_critic.json',communications=configs)
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for train_profile,configuration in configs.items():
  folder=out/f'train_{train_profile}'
  print('Training:',train_profile,flush=True)
  command(['scripts/train_formation.py','--steps','50000','--rollout','512','--epochs','5','--seed','43','--sampling-seed','1000043','--initial-actor',str(initial),'--initial-std','.1','--final-std','.02','--std-decay-steps','20000','--entropy-coef','.001','--gae-lambda','.99','--task','configs/tasks/formation_relative_critic.json','--communication',configuration,'--reward','configs/training/formation_reward_proximity.json','--output',str(folder)],out/f'{train_profile}_training.log')
  for eval_profile,eval_config in configs.items():
   command(['scripts/evaluate_formation.py',str(folder),'--episodes','20','--seed','1000100','--communication',eval_config],out/f'{train_profile}_{eval_profile}_evaluation.log')
   report=json.loads((folder/'evaluation.json').read_text());rows=report['results']
   result=dict(train_profile=train_profile,evaluation_profile=eval_profile,success_rate=report['success_rate'],collisions=sum(r['reason']=='collision' for r in rows),mean_rmse_m=statistics.mean(r['mean_rmse_m'] for r in rows),episodes=len(rows),report=report,path=str(folder))
   results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2))
   print('Result:',json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
 command(['scripts/evaluate_formation.py',str(out/'train_nominal'),'--episodes','20','--seed','1000100','--mode','stationary','--communication',configs['nominal']],out/'stationary.log')
 stationary=json.loads((out/'train_nominal/evaluation.json').read_text())
 hashes=[json.loads((out/f'train_{name}/manifest.json').read_text())['initial_actor_sha256'] for name in configs]
 assert hashes[0]==hashes[1]
 (out/'summary.json').write_text(json.dumps(dict(protocol=protocol,results=results,stationary=stationary),indent=2))
 lines=['# Nominal versus good communication with settling-compatible exploration','',
  'One paired training seed, identical initial actor, 50000 steps per model.',
  'Action standard deviation decreases from 0.1 to 0.02 over 20000 steps.',
  'Twenty common evaluation starts per condition, separate from training.','',
  '| Train communication | Evaluate communication | Success | Collisions | Mean RMSE (m) |',
  '|---|---|---:|---:|---:|']
 for row in results:lines.append(f"| {row['train_profile']} | {row['evaluation_profile']} | {row['success_rate']:.1%} | {row['collisions']} | {row['mean_rmse_m']:.4f} |")
 lines+=['',f"Stationary nominal reference: {stationary['success_rate']:.1%} success.",
  'This is a single-seed development test, not a robust-performance claim.',
  'The formation task excludes five-meter translation and hardware transfer.']
 (root/'docs/good_communication_training_results.md').write_text('\n'.join(lines)+'\n')
 for name in configs:
  folder=out/f'train_{name}'
  trajectory=folder/'trajectory_policy_nominal_1000100.json'
  if trajectory.exists():shutil.copy2(trajectory,folder/'trajectory.json')
  command(['scripts/create_replay.py',str(folder)],out/f'{name}_replay.log')
  command(['scripts/check_actor_export.py',str(folder)],out/f'{name}_export_check.log')
 print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
