"""Mine independent hard starts, fine-tune MAPPO and test on new common starts."""
import argparse,json,shutil,subprocess,sys,statistics,hashlib
from pathlib import Path
from collections import Counter

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);cli=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
    registration=json.loads((root/'experiments/selected_candidate.json').read_text());source=root/registration['run']
    def command(args,path):
        with path.open('w') as log:subprocess.run([sys.executable]+args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    def copyrun(src,dst,filename='checkpoint.pt'):
        dst.mkdir();shutil.copy2(src/'manifest.json',dst/'manifest.json');shutil.copy2(src/filename,dst/'checkpoint.pt')
    baseline=out/'baseline';copyrun(source,baseline)
    protocol=dict(source=str(source),source_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
        training_steps=120000,training_seed=331,sampling_seed=1000331,communication='good',
        selection_seed=22000000,selection_episodes=12,heldout_seed=23000000,heldout_episodes=40,
        broad_seed=24000000,broad_episodes=20,episode_limit_s=120.,
        selection_rule='zero selection collisions; success rate; lower mean RMSE',scope='formation and hold only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Mining fresh failures from frozen current actor.',flush=True)
    command(['learning/evaluation/mine_hard_examples.py','--source',str(baseline),'--output',str(out/'mining')],out/'mining.log')
    print('Focused MAPPO training: 120000 steps under good communication.',flush=True)
    run=out/'training'
    command(['scripts/train_formation.py','--steps','120000','--rollout','512','--epochs','5',
        '--seed','331','--sampling-seed','1000331','--resume',str(baseline/'checkpoint.pt'),
        '--initial-std','.02','--final-std','.02','--std-decay-steps','1',
        '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','30000',
        '--task',str(out/'mining/focused_task.json'),'--communication','configs/communication/good.json',
        '--reward','configs/training/formation_reward_mastery.json','--output',str(run)],out/'training.log')
    def evaluate(folder,profile,seed,count,label,broad=False):
        task='configs/tasks/formation_relational_broad_evaluation.json' if broad else 'configs/tasks/formation_relational_120s_evaluation.json'
        command(['scripts/evaluate_formation.py',str(folder),'--task',task,'--communication',
            f'configs/communication/{profile}.json','--seed',str(seed),'--episodes',str(count)],out/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text());rows=report['results'];success=[x for x in rows if x['reason']=='success']
        result=dict(label=label,path=str(folder),successes=len(success),episodes=len(rows),success_rate=report['success_rate'],
            collisions=sum(x['reason']=='collision' for x in rows),mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),
            median_success_time_s=statistics.median(x['diagnostics']['elapsed_time_s'] for x in success) if success else None,
            failure_stages=dict(Counter(x['diagnostics']['failure_stage'] or x['reason'] for x in rows)),report=report)
        (out/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    choices=[evaluate(baseline,'good',22000000,12,'baseline_selection')]
    for checkpoint in sorted(run.glob('checkpoint_*.pt')):
        folder=out/checkpoint.stem;copyrun(run,folder,checkpoint.name)
        choices.append(evaluate(folder,'good',22000000,12,checkpoint.stem+'_selection'))
    def score(row):return (row['success_rate'],-row['mean_rmse_m'])
    eligible=[x for x in choices if x['collisions']==0]
    selected=Path(max(eligible,key=score)['path'])
    trained=[x for x in eligible if x['path']!=str(baseline)]
    if not trained:raise RuntimeError('No trained checkpoint passed the selection collision gate.')
    best=Path(max(trained,key=score)['path']);results=[]
    for label,folder in [('baseline',baseline),('trained',best)]:
        for profile in ('nominal','good'):results.append(evaluate(folder,profile,23000000,40,label+'_'+profile+'_heldout'))
        results.append(evaluate(folder,'good',24000000,20,label+'_broad_good',broad=True))
    summary=dict(protocol=protocol,selected=str(selected),best_trained=str(best),choices=choices,results=results,
        mining_bank_size=len(json.loads((out/'mining/hard_bank.json').read_text())['entries']),
        training=json.loads((run/'training_summary.json').read_text()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for folder in {selected,best}:
        trajectory=folder/'trajectory_policy_formation_relational_120s_evaluation_v1_good_23000000.json'
        if trajectory.exists():shutil.copy2(trajectory,folder/'trajectory.json')
        command(['scripts/create_replay.py',str(folder)],out/(folder.name+'_replay.log'))
        command(['scripts/check_actor_export.py',str(folder)],out/(folder.name+'_export.log'))
    lines=['# Focused hard-start training results','',
        'The original five inspected failures were not used as training seeds.',
        'New mining pools supplied perturbed hard reset templates; no expert actions or old rollout data were reused.',
        'The policy was fine-tuned for 120000 steps with good communication and unchanged task/reward criteria.',
        'Selection and final tests use disjoint fresh seed ranges. One training seed only.','',
        '| Evaluation | Success | Collisions | Median successful time (s) |',
        '|---|---:|---:|---:|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['median_success_time_s']} |")
    lines+=['',f'Selected before final tests: {selected}',f'Best trained checkpoint: {best}',
        'The five-meter mission and physical robot deployment are not evaluated. No automatic candidate replacement.']
    (root/'docs/hard_focus_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed focused hard-start experiment.',flush=True)

if __name__=='__main__':main()
