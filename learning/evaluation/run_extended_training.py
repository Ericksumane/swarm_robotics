"""Longer-horizon MAPPO continuation with frozen-policy comparisons."""
import argparse,json,shutil,statistics,subprocess,sys
from pathlib import Path
from collections import Counter

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);cli=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=root/'experiments/runs/relational_revision_001/checkpoint_0020480'
    task='configs/tasks/formation_relational_120s_evaluation.json'
    protocol=dict(additional_training_steps=200000,source_checkpoint=str(source/'checkpoint.pt'),
        training_seed=123,sampling_seed=1000123,selection_seed=11000000,selection_episodes=12,
        heldout_seed=12000000,heldout_episodes=40,episode_limit_s=120.,
        training_communication='nominal',geometry='three pair distances of one meter',
        resume_semantics='restore actor, critic and optimizers; fresh episodes and RNG; not exact replay',
        selection_rule='zero selection collisions, highest success, lowest RMSE, lowest successful median duration',
        scope='formation and stationary hold only; not the five-meter mission')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def command(arguments,path):
        with path.open('w') as log:subprocess.run([sys.executable]+arguments,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    def copyrun(src,dst,checkpoint='checkpoint.pt'):
        dst.mkdir();shutil.copy2(src/'manifest.json',dst/'manifest.json');shutil.copy2(src/checkpoint,dst/'checkpoint.pt')
    def evaluate(folder,profile,seed,episodes,label):
        command(['scripts/evaluate_formation.py',str(folder),'--task',task,
            '--communication',f'configs/communication/{profile}.json','--seed',str(seed),'--episodes',str(episodes)],out/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text());rows=report['results']
        success=[x for x in rows if x['reason']=='success']
        result=dict(label=label,path=str(folder),successes=len(success),episodes=len(rows),success_rate=report['success_rate'],
            collisions=sum(x['reason']=='collision' for x in rows),mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),
            median_success_time_s=statistics.median(x['diagnostics']['elapsed_time_s'] for x in success) if success else None,
            mean_abs_speed_mps=statistics.mean(x['diagnostics']['mean_abs_speed_mps'] for x in rows),
            failure_stages=dict(Counter(x['diagnostics']['failure_stage'] or x['reason'] for x in rows)),report=report)
        (out/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    run=out/'training'
    print('Continuing both actor and critic for 200000 steps with 120-second episodes.',flush=True)
    command(['scripts/train_formation.py','--steps','200000','--rollout','512','--epochs','5',
        '--seed','123','--sampling-seed','1000123','--resume',str(source/'checkpoint.pt'),
        '--initial-std','.03','--final-std','.02','--std-decay-steps','40000',
        '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','40000',
        '--task','configs/tasks/formation_relational_120s_training.json',
        '--communication','configs/communication/nominal.json',
        '--reward','configs/training/formation_reward_mastery.json','--output',str(run)],out/'training.log')
    baseline=out/'baseline';copyrun(source,baseline)
    choices=[evaluate(baseline,'nominal',11000000,12,'baseline_selection')]
    for checkpoint in sorted(run.glob('checkpoint_*.pt')):
        folder=out/checkpoint.stem;copyrun(run,folder,checkpoint.name)
        choices.append(evaluate(folder,'nominal',11000000,12,checkpoint.stem+'_selection'))
    def score(row):return (row['success_rate'],-row['mean_rmse_m'],-(row['median_success_time_s'] or float('inf')))
    eligible=[x for x in choices if x['collisions']==0]
    selected=Path(max(eligible,key=score)['path'])
    trained=[x for x in eligible if x['path']!=str(baseline)]
    if not trained:
        (out/'summary.json').write_text(json.dumps(dict(protocol=protocol,choices=choices,selected=str(baseline),reason='No trained checkpoint passed collision gate'),indent=2))
        print('No trained checkpoint passed selection collision gate.',flush=True);return
    best=Path(max(trained,key=score)['path'])
    results=[]
    for label,folder in [('baseline',baseline),('trained',best)]:
        for profile in ('nominal','good'):
            results.append(evaluate(folder,profile,12000000,40,label+'_'+profile+'_heldout'))
    summary=dict(protocol=protocol,selected=str(selected),best_trained=str(best),choices=choices,results=results,
        training=json.loads((run/'training_summary.json').read_text()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for folder in {selected,best}:
        trajectory=folder/'trajectory_policy_formation_relational_120s_evaluation_v1_nominal_12000000.json'
        if trajectory.exists():shutil.copy2(trajectory,folder/'trajectory.json')
        command(['scripts/create_replay.py',str(folder)],out/(folder.name+'_replay.log'))
        command(['scripts/check_actor_export.py',str(folder)],out/(folder.name+'_export.log'))
    lines=['# Extended-horizon training results','',
        'Both the frozen and continued policy are evaluated with the same 120-second limit and common starts.',
        'Training restored actor, critic and optimizers; 200000 new steps, one training seed.',
        'Selection: 12 development seeds. Final evaluation: 40 different common seeds.',
        'Near-formation initial positions with random headings; broad starts and five-meter mission are not tested.','',
        '| Model and communication | Success | Collisions | Median successful duration (s) | Mean RMSE (m) |',
        '|---|---:|---:|---:|---:|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['median_success_time_s']} | {row['mean_rmse_m']:.4f} |")
    lines+=['',f'Selected before final tests: {selected}',f'Best trained checkpoint: {best}',
        'No automatic replacement of the registered research candidate or hardware deployment.',
        'See experiments/runs/extended_horizon_001/summary.json for per-episode diagnostics.']
    (root/'docs/extended_horizon_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed extended-horizon experiment.',flush=True)

if __name__=='__main__':main()
