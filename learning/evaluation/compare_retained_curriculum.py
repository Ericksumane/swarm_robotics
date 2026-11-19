"""Paired on-policy curriculum comparison with fixed actor initialization."""
import argparse,json,shutil,subprocess,sys,statistics
from pathlib import Path
from collections import Counter

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    cli=parser.parse_args();root=Path(__file__).resolve().parents[2]
    out=Path(cli.output).resolve();out.mkdir(parents=True,exist_ok=False)
    initial=root/'experiments/runs/relational_revision_001/checkpoint_0020480'
    actor=initial/'actor.pt'
    protocol=dict(training_seed=79,sampling_seed=1000079,steps_per_arm=60000,
        selection_seed=8000000,selection_episodes=12,heldout_seed=9000000,heldout_episodes=40,
        initial_actor=str(actor),reward='configs/training/formation_reward_mastery.json',
        initial_std=.03,final_std=.02,std_decay_steps=20000,
        selection_rule='zero selection collisions, highest success, lowest mean RMSE; baseline included',
        scope='formation and stationary hold, not translation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def command(arguments,path):
        with path.open('w') as log:subprocess.run([sys.executable]+arguments,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    def evaluate(folder,profile,seed,episodes,label):
        command(['scripts/evaluate_formation.py',str(folder),'--task','configs/tasks/formation_relational_random_evaluation.json',
            '--communication',f'configs/communication/{profile}.json','--seed',str(seed),'--episodes',str(episodes)],out/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text());rows=report['results']
        result=dict(label=label,path=str(folder),successes=sum(x['reason']=='success' for x in rows),episodes=len(rows),
            success_rate=report['success_rate'],collisions=sum(x['reason']=='collision' for x in rows),
            mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),
            failure_stages=dict(Counter(x['diagnostics']['failure_stage'] or x['reason'] for x in rows)),report=report)
        (out/f'{label}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    baseline=out/'baseline';baseline.mkdir()
    for name in ('manifest.json','checkpoint.pt'):shutil.copy2(initial/name,baseline/name)
    selections=[evaluate(baseline,'nominal',8000000,12,'baseline_selection')]
    for name,task in [('legacy','configs/tasks/formation_relational_mastery.json'),('retained','configs/tasks/formation_retained_curriculum.json')]:
        run=out/f'train_{name}'
        print('Training arm:',name,flush=True)
        command(['scripts/train_formation.py','--steps','60000','--rollout','512','--epochs','5',
            '--seed','79','--sampling-seed','1000079','--initial-actor',str(actor),
            '--initial-std','.03','--final-std','.02','--std-decay-steps','20000',
            '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','20000',
            '--task',task,'--communication','configs/communication/nominal.json',
            '--reward','configs/training/formation_reward_mastery.json','--output',str(run)],out/f'{name}_training.log')
        for checkpoint in sorted(run.glob('checkpoint_*.pt')):
            folder=out/f'{name}_{checkpoint.stem}';folder.mkdir()
            shutil.copy2(run/'manifest.json',folder/'manifest.json');shutil.copy2(checkpoint,folder/'checkpoint.pt')
            selections.append(evaluate(folder,'nominal',8000000,12,folder.name+'_selection'))
    hashes=[json.loads((out/f'train_{name}/manifest.json').read_text())['initial_actor_sha256'] for name in ('legacy','retained')]
    assert hashes[0]==hashes[1]
    eligible=[x for x in selections if x['collisions']==0]
    chosen=max(eligible,key=lambda x:(x['success_rate'],-x['mean_rmse_m']))
    selected=Path(chosen['path'])
    print('Selected:',selected.name,flush=True)
    results=[evaluate(baseline,'nominal',9000000,40,'baseline_nominal_heldout')]
    # Evaluate each arm's own selection winner, independent of final test results.
    arm_choices={}
    for arm in ('legacy','retained'):
        candidates=[x for x in selections if Path(x['path']).name.startswith(arm+'_') and x['collisions']==0]
        if candidates:
            choice=max(candidates,key=lambda x:(x['success_rate'],-x['mean_rmse_m']))
            arm_choices[arm]=choice['path']
            results.append(evaluate(Path(choice['path']),'nominal',9000000,40,arm+'_nominal_heldout'))
    results.append(evaluate(selected,'good',9000000,40,'selected_good_heldout'))
    summary=dict(protocol=protocol,selection=selections,selected=str(selected),arm_choices=arm_choices,results=results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    trajectory=selected/'trajectory_policy_formation_relational_random_evaluation_v1_nominal_9000000.json'
    if trajectory.exists():shutil.copy2(trajectory,selected/'trajectory.json')
    command(['scripts/create_replay.py',str(selected)],out/'replay.log')
    command(['scripts/check_actor_export.py',str(selected)],out/'export_check.log')
    lines=['# Retained curriculum comparison','',
        'Two paired 60000-step fine-tunes from the same actor and identical fresh critics.',
        'Reward, exploration schedule and optimizer are matched. Nominal communication only during training.',
        'Selection uses 12 development starts; final tests use 40 different common starts.',
        'Single training seed; near-formation initial geometry. Full five-meter mission is not evaluated.','',
        '| Evaluation | Success | Collisions | Failure diagnostics |',
        '|---|---:|---:|---|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['failure_stages']} |")
    lines+=['',f'Selected before final evaluation: {selected}',
        'No automatic hardware deployment or replacement of the prior candidate.']
    (root/'docs/retained_curriculum_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed. See docs/retained_curriculum_results.md',flush=True)

if __name__=='__main__':main()
