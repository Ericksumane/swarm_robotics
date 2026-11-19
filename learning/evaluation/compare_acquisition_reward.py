"""One-factor reward experiment after diagnosing low-motion acquisition failures."""
import argparse,json,statistics,subprocess,sys,shutil
from pathlib import Path
from collections import Counter

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);cli=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
    initial=root/'experiments/runs/relational_revision_001/checkpoint_0020480'
    def command(args,path):
        with path.open('w') as log:subprocess.run([sys.executable]+args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    def copyrun(source,target):
        target.mkdir()
        for name in ('manifest.json','checkpoint.pt'):shutil.copy2(source/name,target/name)
    def evaluate(folder,profile,seed,episodes,label):
        command(['scripts/evaluate_formation.py',str(folder),'--task','configs/tasks/formation_relational_random_evaluation.json',
            '--communication',f'configs/communication/{profile}.json','--seed',str(seed),'--episodes',str(episodes)],out/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text());rows=report['results']
        failed=[x for x in rows if x['reason']!='success']
        result=dict(label=label,path=str(folder),successes=sum(x['reason']=='success' for x in rows),episodes=len(rows),
            success_rate=report['success_rate'],collisions=sum(x['reason']=='collision' for x in rows),
            mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),
            failure_stages=dict(Counter(x['diagnostics']['failure_stage'] or x['reason'] for x in rows)),
            failed_mean_stopped_s=statistics.mean(x['diagnostics']['stopped_time_s'] for x in failed) if failed else None,report=report)
        (out/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    protocol=dict(training_seed=79,sampling_seed=1000079,training_steps=60000,selection_seed=8000000,
        selection_episodes=12,heldout_seed=10000000,heldout_episodes=40,
        single_changed_factor='motion cost gated by geometric readiness',
        reference='experiments/runs/retained_curriculum_001/train_retained',scope='formation and hold only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run=out/'training'
    command(['scripts/train_formation.py','--steps','60000','--rollout','512','--epochs','5',
        '--seed','79','--sampling-seed','1000079','--initial-actor',str(initial/'actor.pt'),
        '--initial-std','.03','--final-std','.02','--std-decay-steps','20000',
        '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','20000',
        '--task','configs/tasks/formation_retained_curriculum.json',
        '--communication','configs/communication/nominal.json',
        '--reward','configs/training/formation_reward_acquisition.json','--output',str(run)],out/'training.log')
    reference_manifest=json.loads((root/protocol['reference']/'manifest.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    assert reference_manifest['initial_actor_sha256']==manifest['initial_actor_sha256']
    assert reference_manifest['task']==manifest['task']
    baseline=out/'baseline';copyrun(initial,baseline)
    choices=[evaluate(baseline,'nominal',8000000,12,'baseline_selection')]
    for checkpoint in sorted(run.glob('checkpoint_*.pt')):
        folder=out/checkpoint.stem;folder.mkdir();shutil.copy2(run/'manifest.json',folder/'manifest.json');shutil.copy2(checkpoint,folder/'checkpoint.pt')
        choices.append(evaluate(folder,'nominal',8000000,12,checkpoint.stem+'_selection'))
    eligible=[x for x in choices if x['collisions']==0]
    selected=Path(max(eligible,key=lambda x:(x['success_rate'],-x['mean_rmse_m']))['path'])
    new_candidates=[x for x in eligible if x['path']!=str(baseline)]
    new_best=Path(max(new_candidates,key=lambda x:(x['success_rate'],-x['mean_rmse_m']))['path']) if new_candidates else None
    results=[evaluate(baseline,'nominal',10000000,40,'baseline_nominal_heldout')]
    reference_summary=json.loads((root/'experiments/runs/retained_curriculum_001/summary.json').read_text())
    reference=out/'retained_reward_v5';copyrun(Path(reference_summary['arm_choices']['retained']),reference)
    results.append(evaluate(reference,'nominal',10000000,40,'retained_v5_nominal_heldout'))
    if new_best:results.append(evaluate(new_best,'nominal',10000000,40,'acquisition_v6_nominal_heldout'))
    results.append(evaluate(selected,'good',10000000,40,'selected_good_heldout'))
    summary=dict(protocol=protocol,selected=str(selected),choices=choices,results=results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    trajectory=selected/'trajectory_policy_formation_relational_random_evaluation_v1_nominal_10000000.json'
    if trajectory.exists():shutil.copy2(trajectory,selected/'trajectory.json')
    command(['scripts/create_replay.py',str(selected)],out/'replay.log');command(['scripts/check_actor_export.py',str(selected)],out/'export_check.log')
    lines=['# Acquisition reward experiment results','',
        'One-factor comparison: settling motion cost is disabled only outside formation tolerance.',
        'Matching initial actor, training/sampling seeds, fresh critic, task, optimizer and 60000-step budget.',
        'Checkpoint selection used development seeds; final evaluation used 40 new common starts.',
        'One training seed; near-formation positions. No five-meter mission claim.','',
        '| Evaluation | Success | Collisions | Failure types |', '|---|---:|---:|---|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['failure_stages']} |")
    lines+=['',f'Selected before final evaluation: {selected}', 'No automatic deployment or replacement of the current candidate.']
    (root/'docs/acquisition_reward_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed acquisition reward experiment.',flush=True)

if __name__=='__main__':main()
