"""Paired old/global versus relative critic comparison over three seeds."""
import argparse,json,statistics,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy,pack
from swarm_core.formation_env import FormationEnv

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--steps',type=int,default=20000);cli=parser.parse_args()
 root=Path(__file__).resolve().parents[2];out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1)
 configs={'global':'configs/tasks/formation_stage1_geometry.json','relative':'configs/tasks/formation_relative_critic.json'}
 comm=json.loads((root/'configs/communication/nominal.json').read_text());reward=json.loads((root/'configs/training/formation_reward_proximity.json').read_text())
 tasks={name:json.loads((root/path).read_text()) for name,path in configs.items()}
 # Verify the treatment changes critic state only, not actor observations.
 for seed in range(10):
  observations=[]
  for task in tasks.values():
   env=FormationEnv(task,comm,reward);obs,_=env.reset(seed);observations.append(pack(obs))
  np.testing.assert_array_equal(*observations)
 print('Paired actor observations: IDENTICAL',flush=True)
 (out/'protocol.json').write_text(json.dumps(dict(training_seeds=[42,43,44],evaluation_seed_start=900100,evaluation_episodes=20,steps_per_model=cli.steps,task_configs=configs,selection='final checkpoint, no checkpoint tuning'),indent=2))
 results=[]
 def run_command(command,log):
  with log.open('w') as file:subprocess.run(command,cwd=root,stdout=file,stderr=subprocess.STDOUT,check=True)
 for seed in [42,43,44]:
  torch.manual_seed(seed);_,policy,_=make_policy(3,neighbor_dim=9)
  with torch.no_grad():policy.actor.act.action_out.logstd._bias.fill_(float(np.log(.1)))
  initial=out/f'initial_actor_{seed}.pt';torch.save(dict(actor=policy.actor.state_dict()),initial)
  for variant,task_path in configs.items():
   folder=out/f'{variant}_seed_{seed}'
   command=[sys.executable,'scripts/train_formation.py','--steps',str(cli.steps),'--rollout','512','--epochs','5','--seed',str(seed),'--sampling-seed',str(seed+1000000),'--initial-actor',str(initial),'--initial-std','.1','--entropy-coef','.001','--gae-lambda','.99','--task',task_path,'--reward','configs/training/formation_reward_proximity.json','--output',str(folder)]
   print(f'Training {variant}, seed {seed}',flush=True);run_command(command,out/f'{variant}_{seed}_training.log')
   run_command([sys.executable,'scripts/evaluate_formation.py',str(folder),'--episodes','20','--seed','900100'],out/f'{variant}_{seed}_evaluation.log')
   report=json.loads((folder/'evaluation.json').read_text());rows=report['results']
   manifest=json.loads((folder/'manifest.json').read_text())
   result=dict(variant=variant,training_seed=seed,success_rate=report['success_rate'],collisions=sum(r['reason']=='collision' for r in rows),mean_rmse_m=statistics.mean(r['mean_rmse_m'] for r in rows),actor_hash=manifest['initial_actor_sha256'],path=str(folder))
   results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2));print('Result:',json.dumps(result),flush=True)
 for seed in [42,43,44]:
  pair=[r for r in results if r['training_seed']==seed];assert pair[0]['actor_hash']==pair[1]['actor_hash']
 summary={name:dict(mean_success_rate=statistics.mean(r['success_rate'] for r in results if r['variant']==name),mean_rmse_m=statistics.mean(r['mean_rmse_m'] for r in results if r['variant']==name),total_collisions=sum(r['collisions'] for r in results if r['variant']==name)) for name in configs}
 # Stationary reference uses the same 20 starts, not 60 distinct starts.
 reference=Path(results[0]['path'])
 run_command([sys.executable,'scripts/evaluate_formation.py',str(reference),'--episodes','20','--seed','900100','--mode','stationary'],out/'stationary_evaluation.log')
 stationary=json.loads((reference/'evaluation.json').read_text())
 (out/'summary.json').write_text(json.dumps(dict(summary=summary,stationary=stationary,results=results),indent=2))
 lines=['# Paired critic comparison','',
 'Three training seeds; 20000 steps per model; identical initial actor per pair.',
 'Same reward, actor input, sampling seed and evaluation starts. Final checkpoints only.','',
 '| Critic | Mean success rate | Total collisions | Mean RMSE (m) |','|---|---:|---:|---:|']
 for name,data in summary.items():lines.append(f"| {name} | {data['mean_success_rate']:.1%} | {data['total_collisions']} | {data['mean_rmse_m']:.4f} |")
 lines+=['',f"Stationary reference: {stationary['success_rate']:.1%} success; mean RMSE {statistics.mean(r['mean_rmse_m'] for r in stationary['results']):.4f} m.",
 'Each model is tested on the same 20 initial states. These are not 60 independent starts.',
 'Three training seeds are a small development comparison, not a reliability guarantee.',
 'This experiment tests formation acquisition only, not five-meter translation or hardware.']
 (root/'docs/critic_comparison_results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
